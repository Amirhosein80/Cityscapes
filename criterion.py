import torch
import torch.nn as nn
from torch.nn import functional as F


class OhemCE(nn.Module):
    def __init__(self, ignore_label=255, thres=0.9,
                 min_kept=131072, weight=None):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        return self._ohem_forward(score, target)


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super().__init__()
        self.coeff_bce = coeff_bce

    def weighted_bce(self, bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros(log_p.size(), device=log_p.device)
        pos_num = pos_index.sum().to(weight.dtype)
        neg_num = neg_index.sum().to(weight.dtype)
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight = weight.to(log_p.dtype)

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss

    def forward(self, output, target):
        bce_loss = self.weighted_bce(output, target) * self.coeff_bce
        return bce_loss
