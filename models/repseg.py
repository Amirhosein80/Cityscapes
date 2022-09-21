from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchinfo import summary
from torchvision.ops import SqueezeExcitation as SEBlock

from models.fpsparam_calc import fps_calculator
from models.repvgg import RepVGGBlock, repvgg_model_convert, Bottleneck, RepConvN
from models.utils import init_weights, ConvNormAct, IMAGENET_MEAN, IMAGENET_STD, EcaModule
from models.bisenetv2 import GatherExpansion

class ShortCut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                norm_layer=norm_layer)
        self.apply(init_weights)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1,
                               dilation=1, groups=in_channels // 2)
        self.conv3d = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=dilation,
                                dilation=dilation, groups=in_channels // 2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(in_channels * 4))
        self.conv1p = nn.Sequential(nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, 1)
        x1 = self.conv3(x1)
        x2 = self.conv3d(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        return self.conv1p(x)


class WaterfalDilatedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, n: int, use_se=True):
        super().__init__()
        self.cross1 = ConvNormAct(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0,
                                  norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
        self.cross2 = ConvNormAct(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0,
                                  norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(
                DilatedConv(out_channels // 2, out_channels // 2, dilation=dilation))

        self.conv_head = ConvNormAct((len(self.convs) + 2) * (out_channels // 2), out_channels, kernel_size=1,
                                     stride=1, padding=0, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
        if use_se:
            self.atten = EcaModule(out_channels)
        else:
            self.atten = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        res.append(self.cross1(x))
        x = self.cross2(x)
        res.append(x)
        for m in self.convs:
            x = m(x)
            res.append(x)
        x = torch.cat(res, dim=1)
        del res
        x = self.conv_head(x)
        x = self.atten(x)
        return x

class BilateralGuidedAggregation(nn.Module):
    def __init__(self, channel: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.db_dwconv = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1, act=False, groups=channel)
        self.db_pwconv = ConvNormAct(channel, channel, kernel_size=1, stride=1, padding=0)
        self.db_conv2 = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1, act=False)
        self.db_avgpool = nn.AvgPool2d(kernel_size=3, stride=self.scale, padding=1)

        self.sb_dwconv = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1, act=False, groups=channel)
        self.sb_pwconv = ConvNormAct(channel, channel, kernel_size=1, stride=1, padding=0)
        self.sb_conv = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1, act=False)
        self.sb_sigmoid = nn.Sigmoid()

        self.conv_proj = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1)

        self.apply(init_weights)

    def forward(self, db: torch.Tensor, sb: torch.Tensor):
        db_ = self.db_conv2(db)
        db_ = self.db_avgpool(db_)

        db = self.db_dwconv(db)
        db = self.db_pwconv(db)

        sb_ = self.sb_conv(sb)
        sb_ = self.sb_sigmoid(sb_)
        sb_ = f.interpolate(sb_, scale_factor=self.scale, mode="bilinear", align_corners=False)

        sb = self.sb_dwconv(sb)
        sb = self.sb_pwconv(sb)
        sb = self.sb_sigmoid(sb)

        sb = sb * db_
        sb = f.interpolate(sb, scale_factor=self.scale, mode="bilinear", align_corners=False)
        db = db * sb_

        return self.conv_proj(sb + db), db



class RepSegBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = RepVGGBlock(3, 32, stride=2, use_se=False, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(RepVGGBlock(32, 32, stride=2, use_se=False, kernel_size=3, padding=1),
                                    Bottleneck(32, 32),
                                    Bottleneck(32, 32),
                                    Bottleneck(32, 32),
                                    )

        self.layer2 = nn.Sequential(RepVGGBlock(32, 64, stride=2, use_se=False, kernel_size=3, padding=1),
                                    Bottleneck(64, 64), )

        self.layer3 = nn.Sequential(
            RepVGGBlock(64, 256, stride=2, use_se=False, kernel_size=3, padding=1),
            WaterfalDilatedBlock(256, 256, 2, n=1, use_se=True),
            WaterfalDilatedBlock(256, 256, 4, n=4, use_se=True),
        )
        self.layer4 = nn.Sequential(
            WaterfalDilatedBlock(256, 320, 14, n=7, use_se=True),
        )
        self.apply(init_weights)

        self.out_channels = [32, 64, 256, 320]

        self.norm = nn.BatchNorm2d
        self.act = nn.ReLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        outs = OrderedDict()
        x = self.stem(x)
        x = self.layer1(x)
        outs["0"] = x
        x = self.layer2(x)
        outs["1"] = x
        x = self.layer3(x)
        outs["2"] = x
        x = self.layer4(x)
        outs["3"] = x
        return outs


class RepSegDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int, int, int, int], dim: int = 128,
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.conv16 = ConvNormAct(in_channels_list[-1], 128, kernel_size=1, stride=1, padding=0,
                                  norm_layer=norm_layer)
        self.conv8 = ConvNormAct(in_channels_list[1], 128, kernel_size=1, stride=1, padding=0,
                                 norm_layer=norm_layer)
        self.conv4 = RepConvN(in_channels_list[0], dim, kernel_size=3, stride=1, padding=1, act=False)

        self.conv_sum = RepConvN(128, dim, kernel_size=3, stride=1, padding=1, act=False)
        self.bga = BilateralGuidedAggregation(dim, 2)

        self.act = act_layer()
        self.apply(init_weights)

    def forward(self, x: OrderedDict) -> OrderedDict:
        outs = OrderedDict()
        outs["aux1"] = x["0"]
        outs["aux2"] = x["1"]
        outs["aux3"] = x["3"]

        x8 = self.conv8(x["1"])
        x4 = self.conv4(x["0"])
        x = self.conv16(x["3"])

        x = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + x8
        x = self.act(x)

        x = self.conv_sum(x)
        x , x4 = self.bga(sb=x, db=x4)
        outs["edge"] = x4
        outs["out"] = x
        return outs


class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = RepConvN(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.head(x)
        return x


class RepSeg(nn.Module):
    def __init__(self, backbone=None, num_classes: int = 19, dim: int = 64) -> None:
        super().__init__()
        print("RegSeg use default decoder.")
        self.backbone = RepSegBackbone()
        self.decoder = RepSegDecoder(self.backbone.out_channels, dim=dim, norm_layer=self.backbone.norm,
                                     act_layer=self.backbone.act)
        self.classifier = Head(dim, dim, num_classes)
        self.aux1 = Head(self.backbone.out_channels[0], self.backbone.out_channels[0] * 3 // 4, num_classes)
        self.aux2 = Head(self.backbone.out_channels[1], self.backbone.out_channels[1] * 3 // 4, num_classes)
        self.aux3 = Head(self.backbone.out_channels[3], self.backbone.out_channels[3] * 3 // 4, num_classes)
        self.edge = Head(dim, dim // 4, 1)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        _, _, H, W = x.shape
        x = self.backbone(x)
        x = self.decoder(x)
        x["out"] = self.classifier(x["out"])
        x["out"] = f.interpolate(x["out"], size=[H, W], mode='bilinear', align_corners=False)
        if self.training:
            x["aux1"] = self.aux1(x["aux1"])
            x["aux2"] = self.aux2(x["aux2"])
            x["aux3"] = self.aux3(x["aux3"])
            x["edge"] = self.edge(x["edge"])
            x["aux1"] = f.interpolate(x["aux1"], size=[H, W], mode='bilinear', align_corners=False)
            x["aux2"] = f.interpolate(x["aux2"], size=[H, W], mode='bilinear', align_corners=False)
            x["aux3"] = f.interpolate(x["aux3"], size=[H, W], mode='bilinear', align_corners=False)
            x["edge"] = f.interpolate(x["edge"], size=[H, W], mode='bilinear', align_corners=False)
        else:
            x.pop("aux1")
            x.pop("aux2")
            x.pop("aux3")
            x.pop("edge")
        return x

    def get_params(self, lr, weight_decay):
        backbone_wd = []
        backbone_nwd = []
        decoder_wd = []
        decoder_nwd = []
        classifier_wd = []
        classifier_nwd = []
        aux1_wd = []
        aux1_nwd = []
        aux2_wd = []
        aux2_nwd = []
        aux3_wd = []
        aux3_nwd = []
        edge_wd = []
        edge_nwd = []
        for p in self.backbone.parameters():
            if p.dim() == 1:
                backbone_nwd.append(p)
            else:
                backbone_wd.append(p)

        for p in self.decoder.parameters():
            if p.dim() == 1:
                decoder_nwd.append(p)
            else:
                decoder_wd.append(p)

        for p in self.classifier.parameters():
            if p.dim() == 1:
                classifier_nwd.append(p)
            else:
                classifier_wd.append(p)

        for p in self.aux1.parameters():
            if p.dim() == 1:
                aux1_nwd.append(p)
            else:
                aux1_wd.append(p)

        for p in self.aux2.parameters():
            if p.dim() == 1:
                aux2_nwd.append(p)
            else:
                aux2_wd.append(p)

        for p in self.aux3.parameters():
            if p.dim() == 1:
                aux3_nwd.append(p)
            else:
                aux3_wd.append(p)

        for p in self.edge.parameters():
            if p.dim() == 1:
                edge_nwd.append(p)
            else:
                edge_wd.append(p)

        params = [
            {"params": backbone_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_nwd, "lr": lr, "weight_decay": 0},

            {"params": decoder_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": decoder_nwd, "lr": lr, "weight_decay": 0},

            {"params": classifier_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": classifier_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux1_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux1_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux2_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux2_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux3_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux3_nwd, "lr": lr, "weight_decay": 0},

            {"params": edge_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": edge_nwd, "lr": lr, "weight_decay": 0},
        ]
        return params


if __name__ == "__main__":
    x = torch.randn(4, 3, 512, 1024)
    # m = create_RepVGGplus_by_name("RepVGG-A0")
    # m.load_state_dict(torch.load("F:/Amirhosein/AI/Datasets & Code/Cityscapes/RepVGG/RepVGG-A0-train.pth"), strict=False)
    m = RepSeg()
    m = repvgg_model_convert(model=m)
    o = m(x)
    print([(k, v.shape) for k, v in o.items()])
    print(fps_calculator(m, [3, 1024, 2048]))
    print(summary(m, (4, 3, 320, 320)))
    # import timm
    # m = timm.create_model("cs3darknet_focus_s")
    # print(summary(m, (4, 3, 768, 1536)))
