import torch
import torch.nn.functional as f
import torch.nn as nn
import torchvision.ops as ops

from torchinfo import summary
from collections import OrderedDict

from models.utils import ConvNormAct, Head, init_weights, IMAGENET_MEAN, IMAGENET_STD
from models.backbones import backbones
from models.fpsparam_calc import fps_calculator
from models.repvgg import RepVGGBlock, RepConvN, repvgg_model_convert


class DetailBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = RepConvN(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = RepConvN(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = RepConvN(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = RepConvN(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = RepConvN(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = RepConvN(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = RepConvN(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = RepConvN(128, 128, kernel_size=3, stride=1, padding=1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        return x


class ContextEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1
        self.norm = norm_layer(in_channel)
        self.conv1 = ConvNormAct(in_channel, in_channel, kernel_size=1, stride=1, padding=0,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv3 = RepConvN(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.drop = ops.StochasticDepth(p=0.1, mode="row")

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.conv1(self.norm(self.gap(x))))
        x = self.conv3(x)
        return x


class BilateralGuidedAggregation(nn.Module):
    def __init__(self, channel: int, norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        self.db_dwconv = RepConvN(channel, channel, kernel_size=3, stride=1, padding=1, act=False, groups=channel)
        self.db_pwconv = ConvNormAct(channel, channel, kernel_size=1, stride=1, padding=0)
        self.db_conv2 = RepConvN(channel, channel, kernel_size=3, stride=2, padding=1, act=False)
        self.db_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

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
        sb_ = f.interpolate(sb_, scale_factor=4, mode="bilinear", align_corners=False)

        sb = self.sb_dwconv(sb)
        sb = self.sb_pwconv(sb)
        sb = self.sb_sigmoid(sb)

        sb = sb * db_
        sb = f.interpolate(sb, scale_factor=4, mode="bilinear", align_corners=False)
        db = db * sb_

        return self.conv_proj(sb + db), db


class Stem(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv1 = RepConvN(3, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(ConvNormAct(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0,
                                               norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU),
                                   RepConvN(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1))
        self.max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj = RepConvN(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([self.conv2(x), self.max(x)], dim=1)
        x = self.proj(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if stride == 1 and in_channels == out_channels:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Sequential(
                RepConvN(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm_layer=nn.BatchNorm2d)
            )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class GatherExpansion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv3 = RepConvN(in_channels, in_channels * 6, kernel_size=3, stride=1, padding=1)
        self.conv3dw = RepConvN(in_channels * 6, in_channels * 6, kernel_size=3, stride=1, padding=1,
                                groups=in_channels * 6, act=False)
        self.conv1 = ConvNormAct(in_channels * 6, out_channels, kernel_size=1, stride=1, padding=0,
                                 norm_layer=nn.BatchNorm2d)
        self.shortcut = ShortCut(in_channels, out_channels, stride=stride)
        if stride == 1 :
            self.conv3dws = nn.Identity()
        else:
            self.conv3dws = RepConvN(in_channels * 6, in_channels * 6, kernel_size=3, stride=stride, padding=1,
                                     groups=in_channels * 6, act=False)
        self.act = nn.ReLU()
        self.drop = ops.StochasticDepth(p=0.1, mode="row")

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.shortcut(x)
        x = self.conv3(x)
        x = self.conv3dws(x)
        x = self.conv3dw(x)
        x = self.conv1(x)
        x = self.act(self.drop(x) + x_)
        return x

class EdgeExpansion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv3 = RepConvN(in_channels, in_channels * 6, kernel_size=3, stride=stride, padding=1)
        self.conv3dw = RepConvN(in_channels * 6, in_channels * 6, kernel_size=3, stride=1, padding=1,
                                groups=1, act=False)
        self.conv1 = ConvNormAct(in_channels * 6, out_channels, kernel_size=1, stride=1, padding=0,
                                 norm_layer=nn.BatchNorm2d)
        self.shortcut = ShortCut(in_channels, out_channels, stride=stride)

        self.act = nn.ReLU()
        self.drop = ops.StochasticDepth(p=0.1, mode="row")

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.shortcut(x)
        x = self.conv3(x)
        x = self.conv3dw(x)
        x = self.conv1(x)
        x = self.act(self.drop(x) + x_)
        return x

class PAPPM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )

        self.scale_process = RepVGGBlock(mid_channels * 4, mid_channels * 4, kernel_size=3, padding=1, groups=4
                                         , stride=1)

        self.compression = nn.Sequential(
            nn.BatchNorm2d(mid_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(f.interpolate(self.scale1(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(f.interpolate(self.scale2(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(f.interpolate(self.scale3(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(f.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out

class BisenetV2BD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = Stem(16)
        self.layer1 = nn.Sequential(
            EdgeExpansion(16, 32, stride=2),
            EdgeExpansion(32, 32, stride=1),
        )
        self.layer2 = nn.Sequential(
            GatherExpansion(32, 64, stride=2),
            GatherExpansion(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            GatherExpansion(64, 128, stride=2),
            GatherExpansion(128, 128, stride=1),
            GatherExpansion(128, 128, stride=1),
            GatherExpansion(128, 128, stride=1),
        )

        self.bga = BilateralGuidedAggregation(128, norm_layer=nn.BatchNorm2d)
        self.ce = PAPPM(128, 96, 128)
        self.db = nn.Sequential(RepVGGBlock(32, 32, stride=1, kernel_size=3, padding=1),
                                RepConvN(32, 128, stride=1, kernel_size=3, padding=1),)
        # self.db = EdgeExpansion(32, 128, stride=1)
        # self.db = DetailBranch()
        self.apply(init_weights)

        self.out_channels = [16, 32, 64, 128]

        self.norm = nn.BatchNorm2d
        self.act = nn.ReLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor) -> OrderedDict:
        outs = OrderedDict()
        # d = self.db(x)
        x = self.stem(x)
        outs["aux1"] = x
        x = self.layer1(x)
        outs["aux2"] = x
        d = self.db(x)
        x = self.layer2(x)
        outs["aux3"] = x
        x = self.layer3(x)
        outs["aux4"] = x
        x = self.ce(x)
        x, d = self.bga(db=d, sb=x)
        outs["edge"] = d
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


class BisenetV2(nn.Module):
    def __init__(self, backbone=None, num_classes: int = 19, channel: int = 128) -> None:
        super().__init__()
        print("BisenetV2 use default decoder.")
        self.backbone = BisenetV2BD()
        self.classifier = Head(channel, channel, num_classes)
        self.aux1 = Head(16, 16, num_classes)
        self.aux2 = Head(32, 32, num_classes)
        self.aux3 = Head(64, 64, num_classes)
        self.aux4 = Head(128, 128, num_classes)
        self.edge = Head(channel, channel, 1)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        _, _, H, W = x.shape
        x = self.backbone(x)
        x_ = self.classifier(x["out"])
        x["out"] = f.interpolate(x_, size=[H, W], mode='bilinear', align_corners=False)
        if self.training:
            x["aux1"] = self.aux1(x["aux1"])
            x["aux2"] = self.aux2(x["aux2"])
            x["aux3"] = self.aux3(x["aux3"])
            x["aux4"] = self.aux4(x["aux4"])
            x["edge"] = self.edge(x["edge"])
            x["aux1"] = f.interpolate(x["aux1"], size=[H, W], mode='bilinear', align_corners=False)
            x["aux2"] = f.interpolate(x["aux2"], size=[H, W], mode='bilinear', align_corners=False)
            x["aux3"] = f.interpolate(x["aux3"], size=[H, W], mode='bilinear', align_corners=False)
            x["aux4"] = f.interpolate(x["aux4"], size=[H, W], mode='bilinear', align_corners=False)
            x["edge"] = f.interpolate(x["edge"], size=[H, W], mode='bilinear', align_corners=False)
        else:
            x.pop("aux1")
            x.pop("aux2")
            x.pop("aux3")
            x.pop("aux4")
            x.pop("edge")
        return x

    def get_params(self, lr, weight_decay):
        backbone_wd = []
        backbone_nwd = []
        classifier_wd = []
        classifier_nwd = []
        aux1_wd = []
        aux1_nwd = []
        aux2_wd = []
        aux2_nwd = []
        aux3_wd = []
        aux3_nwd = []
        aux4_wd = []
        aux4_nwd = []
        edge_wd = []
        edge_nwd = []

        for p in self.backbone.parameters():
            if p.dim() == 1:
                backbone_nwd.append(p)
            else:
                backbone_wd.append(p)

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

        for p in self.aux4.parameters():
            if p.dim() == 1:
                aux4_nwd.append(p)
            else:
                aux4_wd.append(p)

        for p in self.edge.parameters():
            if p.dim() == 1:
                edge_nwd.append(p)
            else:
                edge_wd.append(p)

        params = [
            {"params": backbone_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_nwd, "lr": lr, "weight_decay": 0},

            {"params": classifier_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": classifier_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux1_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux1_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux2_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux2_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux3_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux3_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux4_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux4_nwd, "lr": lr, "weight_decay": 0},

            {"params": edge_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": edge_nwd, "lr": lr, "weight_decay": 0},
        ]
        return params


if __name__ == "__main__":
    m = BisenetV2()
    i = torch.randn(2, 3, 768, 768)
    m.eval()
    m = repvgg_model_convert(model=m)
    o = m(i)
    print([(k, v.shape) for k, v in o.items()])
    print(summary(m, (16, 3, 768, 768)))
    print(fps_calculator(m, [3, 1024, 2048]))
