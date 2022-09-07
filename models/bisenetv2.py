import torch
import torch.nn.functional as f
import torch.nn as nn

from torchinfo import summary
from collections import OrderedDict

from models.utils import ConvNormAct, Head, AuxHead
from models.backbones import CspDarkNetM
from models.fpsparam_calc import fps_calculator


class DetailBranch(nn.Module):
    def __init__(self, channel: int, norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.conv1_1 = ConvNormAct(3, 64, kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)
        self.conv1_2 = ConvNormAct(64, 64, kernel_size=3, stride=1, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)

        self.conv2_1 = ConvNormAct(64, 64, kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)
        self.conv2_2 = ConvNormAct(64, 64, kernel_size=3, stride=1, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)
        self.conv2_3 = ConvNormAct(64, 64, kernel_size=3, stride=1, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)

        self.conv3_1 = ConvNormAct(64, 128, kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)
        self.conv3_2 = ConvNormAct(128, 128, kernel_size=3, stride=1, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)
        self.conv3_3 = ConvNormAct(128, 128, kernel_size=3, stride=1, padding=1,
                                   norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.conv2_1(x)
        x = self.conv2_3(self.conv2_2(x))

        x = self.conv3_1(x)
        x = self.conv3_3(self.conv3_2(x))

        return x


class ContextEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1
        self.norm = norm_layer(in_channel)
        self.conv1 = ConvNormAct(in_channel, in_channel, kernel_size=1, stride=1, padding=0,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv3 = ConvNormAct(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.alpha = nn.Parameter(torch.ones(in_channel, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (self.alpha * x) + self.conv1(self.norm(self.gap(x)))
        x = self.conv3(x)
        return x


class BilateralGuidedAggregation(nn.Module):
    def __init__(self, channel: int, norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.db_dwconv = ConvNormAct(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.db_pwconv = ConvNormAct(channel, channel, kernel_size=1, stride=1, padding=0)
        self.db_conv2 = ConvNormAct(channel, channel, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)
        self.db_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.sb_dwconv = ConvNormAct(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.sb_pwconv = ConvNormAct(channel, channel, kernel_size=1, stride=1, padding=0)
        self.sb_conv = ConvNormAct(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.sb_sigmoid = nn.Sigmoid()

        self.conv_proj = ConvNormAct(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)

    def forward(self, db: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
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

        return self.conv_proj(sb + db)


class BisenetV2Decoder(nn.Module):
    def __init__(self, sb_channel: int, channel: int, norm_layer: nn.Module = nn.BatchNorm2d,
                 act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.bga = BilateralGuidedAggregation(channel, norm_layer=norm_layer, act_layer=act_layer)
        self.ce = ContextEmbedding(in_channel=sb_channel, out_channel=channel, norm_layer=norm_layer,
                                   act_layer=act_layer)
        self.db = DetailBranch(channel, norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: OrderedDict, img: torch.Tensor) -> OrderedDict:
        outs = OrderedDict()
        db = self.db(img)
        outs["edge"] = db
        outs["aux"] = x["2"]
        sb = self.ce(x["3"])
        outs["out"] = self.bga(sb=sb, db=db)
        return outs


class BisenetV2(nn.Module):
    def __init__(self, backbone: nn.Module = CspDarkNetM, num_classes: int = 19, channel: int = 128) -> None:
        super().__init__()
        self.backbone = backbone()
        self.decoder = BisenetV2Decoder(self.backbone.out_channels[-1], channel, norm_layer=self.backbone.norm,
                                        act_layer=self.backbone.act)
        self.classifier = Head(channel, channel, num_classes,
                               norm_layer=self.backbone.norm, act_layer=self.backbone.act)
        self.aux = AuxHead(self.backbone.out_channels[-2], self.backbone.out_channels[-2] // 4, num_classes,
                           norm_layer=self.backbone.norm, act_layer=self.backbone.act)
        self.edge = AuxHead(channel, channel // 4, 1,
                            norm_layer=self.backbone.norm, act_layer=self.backbone.act)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        _, _, H, W = x.shape
        x = self.decoder(self.backbone(x), x)
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
    m = BisenetV2()
    i = torch.randn(2, 3, 768, 768)
    o = m(i)
    print([(k, v.shape) for k, v in o.items()])
    print(summary(m, (2, 3, 768, 768)))
    print(fps_calculator(m, [3, 768, 1536]))
