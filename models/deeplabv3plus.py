import torch
import torch.nn as nn
import torch.nn.functional as f

from collections import OrderedDict
import torchvision.ops as ops

from models.utils import ConvNormAct, SeparableConv, AuxHead, AttenHead, init_weights
from models.backbones import configs


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1,
                                padding=0, norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x = self.avg(x)
        x = self.conv(x)
        return f.interpolate(x, [H, W], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_list: list[int, int, int] = [6, 12, 18],
                 conv=SeparableConv, norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                norm_layer=norm_layer, act_layer=act_layer)
        self.dil_conv1 = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_list[0],
                              dilation=dilation_list[0], norm_layer=norm_layer, act_layer=act_layer)
        self.dil_conv2 = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_list[1],
                              dilation=dilation_list[1], norm_layer=norm_layer, act_layer=act_layer)
        self.dil_conv3 = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_list[2],
                              dilation=dilation_list[2], norm_layer=norm_layer, act_layer=act_layer)
        self.avg_conv = ASPPPooling(in_channels, out_channels, norm_layer=norm_layer, act_layer=act_layer)
        self.proj = nn.Sequential(ConvNormAct(out_channels * 5, out_channels, kernel_size=1, stride=1,
                                              padding=0, norm_layer=norm_layer, act_layer=act_layer), nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        res = []
        res.append(self.conv(x))
        res.append(self.dil_conv1(x))
        res.append(self.dil_conv2(x))
        res.append(self.dil_conv3(x))
        res.append(self.avg_conv(x))
        res = torch.cat(res, dim=1)
        del x
        return self.proj(res)


class CatAtten(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, norm_layer=norm_layer, act_layer=act_layer)
        self.atten = ops.SqueezeExcitation(input_channels=out_channels, squeeze_channels=out_channels // 4,
                                           activation=act_layer, scale_activation=nn.Sigmoid)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.atten(x)
        return x


class DeeplabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int, int, int, int], dim: int = 256,
                 dilation_list: list[int, int, int] = [6, 12, 18], norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.low_branch = ConvNormAct(in_channels_list[0], 48, kernel_size=1, stride=1, padding=0,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.mid_branch = ConvNormAct(in_channels_list[1], 64, kernel_size=1, stride=1, padding=0,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.aspp = ASPP(in_channels_list[-1], dim, dilation_list=dilation_list, conv=SeparableConv,
                         norm_layer=norm_layer, act_layer=act_layer)
        self.cat_atten = CatAtten(256 + 64, 256, norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: OrderedDict) -> OrderedDict:
        outs = OrderedDict()
        l = self.low_branch(x["0"])
        m = self.mid_branch(x["1"])
        outs["edge"] = x["0"]
        outs["aux"] = x["2"]
        x = self.aspp(x["3"])
        x = f.interpolate(x, size=m.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, m], dim=1)
        x = self.cat_atten(x)
        x = f.interpolate(x, size=l.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, l], dim=1)
        outs["out"] = x
        return outs


class DeeplabV3Plus(nn.Module):
    def __init__(self, backbone, num_classes: int = 19, dim: int = 256) -> None:
        super().__init__()
        try:
            self.backbone = backbone(output_stride=16)
        except:
            raise ValueError("Your Backbone doesn't support dilation.")
        self.decoder = DeeplabV3PlusDecoder(self.backbone.out_channels, dim=dim, dilation_list=[6, 12, 18],
                                            norm_layer=self.backbone.norm, act_layer=self.backbone.act)
        self.classifier = AttenHead(dim + 48, dim, num_classes, norm_layer=self.backbone.norm,
                                    act_layer=self.backbone.act)
        self.aux = AuxHead(self.backbone.out_channels[-2], self.backbone.out_channels[-2] // 4, num_classes,
                           norm_layer=self.backbone.norm, act_layer=self.backbone.act)
        self.edge = AuxHead(self.backbone.out_channels[0], self.backbone.out_channels[0] // 4, 1,
                            norm_layer=self.backbone.norm, act_layer=self.backbone.act)

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
    m = DeeplabV3Plus(backbone=configs["repvgg"])
    i = torch.randn(2, 3, 768, 1536)
    o = m(i)
