import torch
import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict
from torchvision.ops import SqueezeExcitation
from torchinfo import summary

from models.utils import ConvNormAct, IMAGENET_MEAN, IMAGENET_STD, init_weights
from models.fpsparam_calc import fps_calculator
from models.repvgg import RepVGGBlock, repvgg_model_convert

class ShortCut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        if stride == 1:
            self.avg = nn.Identity()
        else:
            self.avg = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg(x)
        x = self.conv(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups_width: int = 16,
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        groups = out_channels // groups_width
        self.conv_first = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.conv_mid = ConvNormAct(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                    groups=groups, norm_layer=norm_layer, act_layer=act_layer)
        self.conv_end = ConvNormAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                    norm_layer=norm_layer)
        self.act = act_layer()
        self.atten = SqueezeExcitation(out_channels, out_channels // 4, activation=act_layer)
        self.shortcut = ShortCut(in_channels, out_channels, stride=stride, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.conv_first(x)
        x_ = self.conv_mid(x_)
        x_ = self.atten(x_)
        x_ = self.conv_end(x_)
        x = self.shortcut(x)
        return self.act(x + x_)


class DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups_width: int = 16,
                 dilations: list = [1, 10],
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.dilations = dilations
        groups = out_channels // groups_width // len(self.dilations)
        self.conv_first = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.conv_mid = []
        for d in dilations:
            self.conv_mid.append(
                ConvNormAct(out_channels // len(self.dilations), out_channels // len(self.dilations), kernel_size=3,
                            stride=stride, padding=d,
                            dilation=d, groups=groups, norm_layer=norm_layer, act_layer=act_layer))
        self.conv_mid = nn.ModuleList(self.conv_mid)
        self.conv_end = ConvNormAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                    norm_layer=norm_layer)
        self.act = act_layer()
        self.atten = SqueezeExcitation(out_channels, out_channels // 4, activation=act_layer)
        self.shortcut = ShortCut(in_channels, out_channels, stride=stride, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.conv_first(x)
        x_ = torch.chunk(x_, len(self.dilations), dim=1)
        x_ = list(x_)
        for i in range(len(self.dilations)):
            x_[i] = self.conv_mid[i](x_[i])
        x_ = torch.cat(x_, dim=1)
        # x_ = self.conv_mid(x_)
        x_ = self.atten(x_)
        x_ = self.conv_end(x_)
        x = self.shortcut(x)
        return self.act(x + x_)


class RegSegBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, 32, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU)
        self.layer1 = nn.Sequential(YBlock(32, 48, stride=2))

        self.layer2 = nn.Sequential()
        for i in range(2):
            self.layer2.append(YBlock(48, 48, stride=1,))
        self.layer2.append(YBlock(48, 128, stride=2))

        self.layer3 = nn.Sequential()
        for i in range(1):
            self.layer3.append(YBlock(128, 128, stride=1))
        self.layer3.append(YBlock(128, 256, stride=2))

        self.layer4 = nn.Sequential(
            DBlock(256, 256, stride=1, dilations=[1, 2], norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU))
        for i in range(4):
            self.layer4.append(
                DBlock(256, 256, stride=1, dilations=[1, 4], norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU))
        for i in range(6):
            self.layer4.append(
                DBlock(256, 256, stride=1, dilations=[1, 14], norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU))
        self.layer4.append(DBlock(256, 320, stride=1, dilations=[1, 14], norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU))

        self.out_channels = [48, 128, 256, 320]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
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


class RegSegDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int, int, int, int], dim: int = 64,
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.conv16 = ConvNormAct(in_channels_list[-1], 128, kernel_size=1, stride=1, padding=0,
                                  norm_layer=norm_layer, act_layer=act_layer)
        self.conv8 = ConvNormAct(in_channels_list[1], 128, kernel_size=1, stride=1, padding=0,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv4 = ConvNormAct(in_channels_list[0], 8, kernel_size=1, stride=1, padding=0,
                                 norm_layer=norm_layer, act_layer=act_layer)

        self.conv_sum = RepVGGBlock(128, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: OrderedDict) -> OrderedDict:
        outs = OrderedDict()
        outs["edge"] = x["0"]
        outs["aux"] = x["2"]

        x8 = self.conv8(x["1"])
        x4 = self.conv4(x["0"])

        x = self.conv16(x["3"])
        x = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + x8

        x = self.conv_sum(x)
        x = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x4], dim=1)
        outs["out"] = x
        return outs


class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = RepVGGBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.head(x)
        return x


class RegSeg(nn.Module):
    def __init__(self, backbone, num_classes: int = 19, dim: int = 64) -> None:
        super().__init__()
        print("RegSeg use default decoder.")
        self.backbone = RegSegBackbone()
        self.decoder = RegSegDecoder(self.backbone.out_channels, dim=dim, norm_layer=self.backbone.norm,
                                     act_layer=self.backbone.act)
        self.classifier = Head(dim + 8, dim, num_classes)
        self.aux = Head(self.backbone.out_channels[-2], self.backbone.out_channels[-2] // 4, num_classes)
        self.edge = Head(self.backbone.out_channels[0], self.backbone.out_channels[0], 1)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        _, _, H, W = x.shape
        x = self.backbone(x)
        x = self.decoder(x)
        x["out"] = self.classifier(x["out"])
        x["aux"] = self.aux(x["aux"])
        x["edge"] = self.edge(x["edge"])
        x["out"] = f.interpolate(x["out"], size=[H, W], mode='bilinear', align_corners=False)
        x["edge"] = f.interpolate(x["edge"], size=[H, W], mode='bilinear', align_corners=False)
        x["aux"] = f.interpolate(x["aux"], size=[H, W], mode='bilinear', align_corners=False)
        return x

    def get_params(self, lr, weight_decay):
        backbone_wd = []
        backbone_nwd = []
        decoder_wd = []
        decoder_nwd = []
        classifier_wd = []
        classifier_nwd = []
        aux_wd = []
        aux_nwd = []
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

        for p in self.aux.parameters():
            if p.dim() == 1:
                aux_nwd.append(p)
            else:
                aux_wd.append(p)

        for p in self.edge.parameters():
            if p.dim() == 1:
                edge_nwd.append(p)
            else:
                edge_wd.append(p)

        params = [
            {"params": backbone_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_nwd, "lr": 0.1 * lr, "weight_decay": 0},

            {"params": decoder_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": decoder_nwd, "lr": lr, "weight_decay": 0},

            {"params": classifier_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": classifier_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux_nwd, "lr": lr, "weight_decay": 0},

            {"params": edge_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": edge_nwd, "lr": lr, "weight_decay": 0},
        ]
        return params

     


if __name__ == "__main__":
    x = torch.randn(4, 3, 768, 768)
    m = RegSeg(backbone=None)
    m = repvgg_model_convert(model=m)
    o = m(x)
    print([(k, v.shape) for k, v in o.items()])
    print(fps_calculator(m, [3, 768, 1536]))
    print(summary(m, (4, 3, 768, 768)))
