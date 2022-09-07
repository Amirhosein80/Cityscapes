import torch
import torch.nn as nn
import torch.nn.functional as f


# code from https://github.com/MichaelFan01/STDC-Seg.git
class EdegeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).requires_grad_(False)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.conv.weight = nn.Parameter(torch.tensor(
                [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                dtype=torch.float32).reshape(1, 1, 3, 3), requires_grad=False)
            self.fuse.weight = nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                         dtype=torch.float32).reshape(1, 3, 1, 1), requires_grad=False)

    def forward(self, x: torch.Tensor):
        device = x.device
        x = x.float().to(device)
        self.conv.stride = (1, 1)
        boundary_targets = self.conv(x)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        self.conv.stride = (2, 2)
        boundary_targets_x2 = self.conv(x)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        self.conv.stride = (4, 4)
        boundary_targets_x4 = self.conv(x)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x4_up = f.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = f.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = self.fuse(boudary_targets_pyramids)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        return boudary_targets_pyramid


class Target2Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = EdegeDetector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.edge.eval()
        with torch.no_grad():
            x = torch.unsqueeze(x, dim=1)
            return self.edge(x).squeeze(1).long()


if __name__ == "__main__":
    pass
