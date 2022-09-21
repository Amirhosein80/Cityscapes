import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os

from criterion import OhemCrossEntropy, BoundaryLoss, OhemCE


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def poly_lr_scheduler(current_iter, total_iters,warmup_iters,warmup_factor,p=0.9):
    lr=(1 - current_iter / total_iters) ** p
    if current_iter<warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr

def get_lr_function(total_iterations, warmup_iters, warmup_factor, power):
    return lambda x : poly_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,power)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metric(nn.Module):
    def __init__(self, num_classes: int, mode="multiclass") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

    def forward(self, output, target):
        tp, fp, fn, tn = smp.metrics.get_stats(output=output, target=target, mode=self.mode,
                                               num_classes=self.num_classes)
        return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


class SegWithAuxLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, weight=None) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = OhemCE(ignore_label=ignore_index, weight=weight)
        self.edgeloss = BoundaryLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, target, edge_targets):
        semantic_loss = 0
        semantic_aux = 0
        edge_loss = 0
        loss_sb = 0
        for k, v in outputs.items():
            if "out" in k:
                if "edge" in list(outputs.keys()):
                    filler = torch.ones_like(target) * self.ignore_index
                    bd_label = torch.where(self.sigmoid(outputs["edge"][:, 0, :, :]) > 0.8, target, filler)
                    loss_sb = self.loss(v, bd_label)
                semantic_loss += self.loss(v, target) + loss_sb
            elif "aux" in k:
                semantic_aux += self.loss(v, target)
            elif "edge" in k:
                edge_loss += self.edgeloss(v, edge_targets)
        return semantic_loss, semantic_aux, edge_loss


class Checkpoint(object):
    def __init__(self, best_acc):
        self.best_acc = best_acc
        self.folder = './checkpoint'
        os.makedirs(self.folder, exist_ok=True)

    def save(self, model, acc, filename, scaler, optimizer, scheduler, epoch=-1):
        if acc > self.best_acc:
            print('Saving checkpoint...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'scaler': scaler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            path = os.path.join(os.path.abspath(self.folder), f"{filename}_{epoch}" + '.pth')
            torch.save(state, path)
            torch.save(state, "./best.pth")
            self.best_acc = acc


