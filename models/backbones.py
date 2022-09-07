import torch
import torch.nn as nn
import timm

from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter
from torchinfo import summary

from models.utils import dilation2, IMAGENET_MEAN, IMAGENET_STD, MOBILENET_MEAN, MOBILENET_STD


# ==============================================
#  Dilation Backbones (Support stride 16 or 32)
# ==============================================

class MobileNetV3(nn.Module):
    def __init__(self, output_stride=32) -> None:
        super().__init__()
        assert output_stride == 32 or output_stride == 16, "Only support stride 16 or 32"
        model = timm.create_model("mobilenetv3_large_100_miil", pretrained=True)
        self.stem = model.conv_stem
        self.stages = model.blocks[:-1]
        del model

        if output_stride == 16:
            self.stages[-1].apply(dilation2)

        self.stages = IntermediateLayerGetter(self.stages, return_layers={"1": "0",
                                                                          "2": "1",
                                                                          "4": "2",
                                                                          "5": "3", })
        self.out_channels = [24, 40, 112, 160]

        self.act = nn.Hardswish
        self.norm = nn.BatchNorm2d
        self.mean = MOBILENET_MEAN
        self.std = MOBILENET_STD

    def forward(self, x: torch.Tensor) -> OrderedDict:
        x = self.stem(x)
        x = self.stages(x)
        return x


class CspDarkNetM(nn.Module):
    def __init__(self, output_stride=32) -> None:
        super().__init__()
        assert output_stride == 32 or output_stride == 16, "Only support stride 16 or 32"

        model = timm.create_model("cs3darknet_focus_m", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages

        del model
        if output_stride == 16:
            self.stages[-1].apply(dilation2)
        self.stages = IntermediateLayerGetter(self.stages, return_layers={"0": "0",
                                                                          "1": "1",
                                                                          "2": "2",
                                                                          "3": "3", })
        self.out_channels = [96, 192, 384, 768]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


class CspDarkNetL(nn.Module):
    def __init__(self, output_stride=32) -> None:
        super().__init__()
        assert output_stride == 32 or output_stride == 16, "Only support stride 16 or 32"

        model = timm.create_model("cs3darknet_focus_l", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages

        del model
        if output_stride == 16:
            self.stages[-1].apply(dilation2)
        self.stages = IntermediateLayerGetter(self.stages, return_layers={"0": "0",
                                                                          "1": "1",
                                                                          "2": "2",
                                                                          "3": "3", })
        self.out_channels = [128, 256, 512, 1024]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


class RepVGG(nn.Module):
    def __init__(self, output_stride=32) -> None:
        super().__init__()
        assert output_stride == 32 or output_stride == 16, "Only support stride 16 or 32"

        model = timm.create_model("repvgg_b1g4", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages

        del model
        if output_stride == 16:
            self.stages[-1].apply(dilation2)
        self.stages = IntermediateLayerGetter(self.stages, return_layers={"0": "0",
                                                                          "1": "1",
                                                                          "2": "2",
                                                                          "3": "3", })
        self.out_channels = [128, 256, 512, 2048]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


# ==========================================
#  Fixed Backbones (Only support stride 32)
# ==========================================

class EdgeNext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = timm.create_model("edgenext_small", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages
        del model

        self.stages = IntermediateLayerGetter(self.stages, return_layers={"0": "0",
                                                                          "1": "1",
                                                                          "2": "2",
                                                                          "3": "3", })
        self.out_channels = [48, 96, 160, 304]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


class MobileVitV2200(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = timm.create_model("mobilevitv2_200_in22ft1k", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages
        del model

        self.stages = IntermediateLayerGetter(self.stages, return_layers={"1": "0",
                                                                          "2": "1",
                                                                          "3": "2",
                                                                          "4": "3", })
        self.out_channels = [256, 512, 768, 1024]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


class MobileVitV2150(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = timm.create_model("mobilevitv2_150_in22ft1k", pretrained=True)

        self.stem = model.stem
        self.stages = model.stages
        del model

        self.stages = IntermediateLayerGetter(self.stages, return_layers={"1": "0",
                                                                          "2": "1",
                                                                          "3": "2",
                                                                          "4": "3", })
        self.out_channels = [192, 384, 576, 768]

        self.norm = nn.BatchNorm2d
        self.act = nn.SiLU
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x


configs = {
    "mobilenetv3": MobileNetV3,
    "cspdarknetm": CspDarkNetM,
    "cspdarknetl": CspDarkNetL,
    "repvgg": RepVGG,
    "edgenext": EdgeNext,
    "mobilevitv2150": MobileVitV2150,
    "mobilevitv2200": MobileVitV2200,
}

if __name__ == "__main__":
    m = CspDarkNetM(16)
    i = torch.randn(2, 3, 512, 512)
    o = m(i)
    # print([(k, v.shape) for k, v in o.items()])
    print(summary(m, (2, 3, 512, 512)))
