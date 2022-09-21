import torch
import torch.nn as nn
import torch.nn.functional as f

from models.utils import init_weights, ConvNormAct, IMAGENET_MEAN, IMAGENET_STD, EcaModule
from models.repvgg import RepVGGBlock, repvgg_model_convert
from collections import OrderedDict
from torchinfo import summary
from models.fpsparam_calc import fps_calculator


class ShortCut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        if in_channels == out_channels:
            self.conv = nn.Identity()
        else:
            self.conv = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                    norm_layer=norm_layer, act_layer=nn.ReLU)
        if stride == 1:
            self.avg = nn.Identity()
        else:
            self.avg = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg(x)
        x = self.conv(x)
        return x


class VovBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int, stride: int = 1, act=True):
        super().__init__()
        if in_channels == out_channels:
            self.short_cut = nn.Identity()
            self.cross = nn.Identity()
        else:
            self.short_cut = ShortCut(in_channels, out_channels, stride=stride)
            self.cross = ShortCut(in_channels, out_channels, stride=stride)
        self.convs = nn.ModuleList()
        self.convs.append(
            RepVGGBlock(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, use_se=False))
        for i in range(n):
            self.convs.append(
                RepVGGBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, use_se=False))

        self.conv_head = ConvNormAct(len(self.convs) * out_channels // 2 + out_channels, out_channels, kernel_size=1,
                                     stride=1, padding=0, norm_layer=nn.BatchNorm2d)
        self.atten = EcaModule(channels=out_channels)
        self.alphas = nn.Parameter(torch.ones(1, out_channels, 1, 1), requires_grad=True)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.short_cut(x)
        res = [self.act(self.cross(x))]
        for m in self.convs:
            x = m(x)
            res.append(x)
        x = torch.cat(res, dim=1)
        del res
        x = self.conv_head(x)
        x = self.atten(x)
        x = x + (x_ * self.alphas)
        return self.act(x)


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_s = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                  norm_layer=nn.BatchNorm2d)

        self.conv_d = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                  norm_layer=nn.BatchNorm2d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, d, atten):
        _, _, H, W = d.shape
        atten = self.sigmoid(atten)
        s = f.interpolate(s, size=[H, W],
                          mode='bilinear', align_corners=False)
        d_add = self.conv_d((1 - atten) * s + d)
        s_add = self.conv_s(s + atten * d)
        return d_add + s_add


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


class BilateralFusion(nn.Module):
    def __init__(self, semantic_channel, detail_channel, scale):
        super().__init__()
        self.scale = scale
        self.semantic = ShortCut(semantic_channel, detail_channel)
        self.detail = RepVGGBlock(detail_channel, semantic_channel, kernel_size=3, stride=scale, padding=1,
                                  act=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, s: torch.Tensor, d: torch.Tensor):
        d_ = self.semantic(s)
        s_ = self.detail(d)
        d_ = f.interpolate(d_, scale_factor=self.scale, mode="bilinear", align_corners=False)
        d = self.act(d + d_)
        s = self.act(s + s_)
        return s, d


class RepSegV2Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = RepVGGBlock(3, 32, stride=2, use_se=False, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(RepVGGBlock(32, 32, stride=2, use_se=False, act=True,
                                                kernel_size=3, padding=1),
                                    ResRep(32, 32),
                                    ResRep(32, 32))

        self.layer2 = nn.Sequential(RepVGGBlock(32, 64, stride=2, use_se=False, act=True,
                                                kernel_size=3, padding=1),
                                    ResRep(64, 64), )

        self.layer3 = nn.Sequential(RepVGGBlock(64, 128, stride=2, use_se=False, act=True,
                                                kernel_size=3, padding=1),
                                    ResRep(128, 128, act=False), )
        self.layer4 = nn.Sequential(RepVGGBlock(128, 256, stride=2, use_se=False, act=True,
                                                kernel_size=3, padding=1),
                                    ResRep(256, 256, act=False), )

        self.layer5 = VovBlock(256, 512, stride=2, n=0)

        self.layer3_d = nn.Sequential(RepVGGBlock(64, 64, stride=1, use_se=False, act=True,
                                                  kernel_size=3, padding=1),
                                      ResRep(64, 64, act=False), )

        self.layer4_d = nn.Sequential(RepVGGBlock(64, 64, stride=1, use_se=False, act=True,
                                                  kernel_size=3, padding=1),
                                      ResRep(64, 64, act=False), )

        self.layer5_d = VovBlock(64, 128, stride=1, n=0)

        self.layer4_a = ResRep(64, 64)

        self.layer5_a = VovBlock(64, 128, stride=1, n=0)

        self.bf1 = BilateralFusion(semantic_channel=128, detail_channel=64, scale=2)
        self.bf2 = BilateralFusion(semantic_channel=256, detail_channel=64, scale=4)

        self.atten_add = ShortCut(256, 64)
        self.act = nn.ReLU(inplace=True)

        self.bag = Bag(128, 64)
        self.ppm = PAPPM(512, 96, 128)

        self.out_channels = [64, 128, 256, 512]

        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        outs = OrderedDict()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        d = self.layer3_d(x)
        x = self.layer3(x)
        x, d = self.bf1(s=x, d=d)
        outs["aux1"] = d
        outs["aux2"] = x
        d = self.layer4_d(d)
        x = self.layer4(x)
        x, d = self.bf2(s=x, d=d)
        d = self.layer5_d(d)
        x = self.layer5(x)
        x = self.ppm(x)
        outs["out"] = d + f.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)
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


class RepSegV2(nn.Module):
    def __init__(self, backbone=None, num_classes: int = 19, dim: int = 64) -> None:
        super().__init__()
        print("RegSeg use default decoder.")
        self.backbone = RepSegV2Backbone()
        self.decoder = RepSegV2Decoder(self.backbone.out_channels, dim=dim)
        self.classifier = Head(dim * 2, dim, num_classes)
        self.aux = Head(dim, dim, num_classes)
        self.edge = Head(dim, dim, 1)

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
            {"params": backbone_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_nwd, "lr": lr, "weight_decay": 0},

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


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


if __name__ == "__main__":
    x = torch.randn(4, 3, 768, 1536)
    # x = torch.randn(4, 512, 12, 12)
    m = RepSegV2Backbone()
    # m = PAPPM(512, 96, 128)
    # m = repvgg_model_convert(model=m)
    o = m(x)
    print([(k, v.shape) for k, v in o.items()])
    # print(fps_calculator(m, [512, 12, 12]))
    # print(summary(m, (4, 512, 12, 12)))
    print(fps_calculator(m, [3, 1024, 2048]))
    print(summary(m, (4, 3, 768, 1536)))
