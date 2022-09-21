import torch
import torch.nn as nn
import torch.nn.functional as f
import torchinfo

from models.se_block import SEBlock
from collections import OrderedDict

from models.fpsparam_calc import fps_calculator
from models.utils import init_weights, ConvNormAct
from models.backbones import backbones
from models.repvgg import RepConvN

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: nn.Module = None) -> None:
        super(AttentionRefinementModule, self).__init__()
        self.conv = RepConvN(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_atten = ConvNormAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      norm_layer=norm_layer)
        self.sigmoid_atten = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        atten = self.avg(x)
        atten = self.conv_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(x, atten)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = RepConvN(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.atten = SEBlock(out_channels, out_channels // 16)
        self.apply(init_weights)
        self.atten.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.atten(x)
        return x


class BisenetDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int, int, int, int], dim: int = 128,
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.arm32 = AttentionRefinementModule(in_channels_list[-1], dim, norm_layer=norm_layer)
        self.arm16 = AttentionRefinementModule(in_channels_list[-2], dim, norm_layer=norm_layer)
        self.conv_head32 = RepConvN(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = RepConvN(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_avg = ConvNormAct(in_channels_list[-1], dim, kernel_size=1, stride=1, padding=0,
                                    norm_layer=norm_layer, act_layer=act_layer)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FeatureFusionModule(in_channels_list[1] + dim, dim * 2)

    def forward(self, x: OrderedDict) -> OrderedDict:
        outs = OrderedDict()
        outs["edge"] = x["1"]
        outs["aux"] = x["2"]
        _, _, H32, W32 = x["3"].shape
        _, _, H16, W16 = x["2"].shape
        _, _, H8, W8 = x["1"].shape
        avg = self.avg(x["3"])
        avg = self.conv_avg(avg)
        avg = f.interpolate(avg, [H32, W32], mode="nearest")

        x["3"] = self.arm32(x["3"])
        x["3"] = x["3"] + avg
        x["3"] = f.interpolate(x["3"], (H16, W16), mode='nearest')
        x["3"] = self.conv_head32(x["3"])

        x["2"] = self.arm16(x["2"])
        x["2"] = x["2"] + x["3"]
        x["2"] = f.interpolate(x["2"], (H8, W8), mode='nearest')
        x["2"] = self.conv_head16(x["2"])

        x = torch.cat([x["1"], x["2"]], dim=1)
        x = self.ffm(x)
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

class Bisenet(nn.Module):
    def __init__(self, backbone, num_classes: int = 19, dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone()
        self.decoder = BisenetDecoder(self.backbone.out_channels, dim=dim)
        self.classifier = Head(dim * 2, dim * 2, num_classes)
        self.aux = Head(self.backbone.out_channels[-2], self.backbone.out_channels[-2] // 4, num_classes)
        self.edge = Head(self.backbone.out_channels[1], self.backbone.out_channels[1] // 4, 1)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        _, _, H, W = x.shape
        x = self.backbone(x)
        x = self.decoder(x)
        x["out"] = self.classifier(x["out"])
        x["out"] = f.interpolate(x["out"], size=[H, W], mode='bilinear', align_corners=False)
        if self.training:
            x["aux"] = self.aux(x["aux"])
            x["edge"] = self.edge(x["edge"])
            x["aux"] = f.interpolate(x["aux"], size=[H, W], mode='bilinear', align_corners=False)
            x["edge"] = f.interpolate(x["edge"], size=[H, W], mode='bilinear', align_corners=False)
        else:
            x.pop("aux")
            x.pop("edge")
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
            {"params": backbone_wd, "lr": 0.1 * lr, "weight_decay": weight_decay},
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
    m = Bisenet(backbones["repvgg"])
    i = torch.randn(2, 3, 768, 768)
    o = m(i)
    print(torchinfo.summary(m, (2, 3, 768, 768)))
    print([(k, v.shape) for k, v in o.items()])
    print(fps_calculator(m, [3, 1024, 2048]))
