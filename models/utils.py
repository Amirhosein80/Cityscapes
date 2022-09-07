import torch
import torch.nn as nn
import torchvision.ops as ops

# ============
#  Constants
# =============

XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD = [0.5, 0.5, 0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MOBILENET_MEAN = [0.0, 0.0, 0.0]
MOBILENET_STD = [1.0, 1.0, 1.0]


# ============
#  Functions
# =============

def dilation2(m):
    if isinstance(m, nn.Conv2d):
        if m.stride == (2, 2):
            m.stride = (1, 1)

        else:
            if m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)
            elif m.kernel_size == (5, 5):
                m.dilation = (2, 2)
                m.padding = (4, 4)
            elif m.kernel_size == (7, 7):
                m.dilation = (2, 2)
                m.padding = (6, 6)
            elif m.kernel_size == (9, 9):
                m.dilation = (2, 2)
                m.padding = (8, 8)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


# ============
#  Classes
# =============

class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                               padding=padding, bias=True if norm_layer is None else False,
                                               dilation=dilation))
        if norm_layer is not None:
            self.conv.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            try:
                self.conv.add_module("act", act_layer(inplace=True))
            except:
                self.conv.add_module("act", act_layer())

        self.conv.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_dw", nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                                  padding=padding, bias=False,
                                                  dilation=dilation, groups=in_channels))
        self.conv.add_module("conv_sp", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                                  padding=0, bias=True if norm_layer is None else False))
        if norm_layer is not None:
            self.conv.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            try:
                self.conv.add_module("act", act_layer(inplace=True))
            except:
                self.conv.add_module("act", act_layer())

        self.conv.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, norm_layer=norm_layer, act_layer=act_layer)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.head(x)
        return x


class AttenHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=5, stride=1,
                                padding=2, norm_layer=norm_layer, act_layer=act_layer)
        self.atten = ops.SqueezeExcitation(input_channels=out_channels, squeeze_channels=out_channels // 4,
                                           activation=act_layer, scale_activation=nn.Sigmoid)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.atten(x)
        x = self.head(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int, norm_layer: nn.Module = None,
                 act_layer: nn.Module = None) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, norm_layer=norm_layer, act_layer=act_layer)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.drop(x)
        x = self.head(x)
        return x
