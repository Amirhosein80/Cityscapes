import argparse
import datetime
import tensorboardX as ts
import torch
import gc
import torch.optim as optim
import torch.cuda.amp as amp
import tqdm.autonotebook as tqdm
import os

from tabulate import tabulate
from torchinfo import summary
from colorama import Fore
from torch.utils.data import DataLoader

from cityscapes import CityScapes, cityscapes_config, CutMix
from models.deeplabv3plus import DeeplabV3Plus
from models.bisenetv2 import BisenetV2
from models.bisenet import Bisenet
from models.segformer import SegFormer
from models.backbones import backbones
from models.regseg import RegSeg
from models.repseg import RepSeg
from models.repsegv2 import RepSegV2
from models.edge import Target2Edge
from utils import AverageMeter, Metric, SegWithAuxLoss, Checkpoint, get_lr, count_parameters, get_lr_function

gc.collect()
torch.cuda.empty_cache()
now = datetime.datetime.now()
now = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
device = "cuda" if torch.cuda.is_available() else "cpu"
writer = ts.SummaryWriter(f"runs/CityscapeModels{now}")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
name = "CityscapeModels"

models = {"deeplabv3plus": DeeplabV3Plus,
          "bisenetv2": BisenetV2,
          "segformer": SegFormer,
          "bisenet": Bisenet,
          "regseg": RegSeg,
          "repseg": RepSeg,
          "repsegv2": RepSegV2,
          }


def get_args():
    parser = argparse.ArgumentParser(description="RTS Segmentation Training", add_help=True)
    parser.add_argument("--backbone", default="cspdarknetl", type=str,
                        help=f"backbone name (it's should be on of {tuple(backbones.keys())}, default: repvgg)")
    parser.add_argument("--model", default="repseg", type=str,
                        help=f"model name (it's should be on of {tuple(models.keys())}, default: deeplabv3plus)")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=500, type=int, help="number of total epochs to run")
    parser.add_argument("--warmup-iter", default=3000, type=int, help="number of iter for warmup model")
    parser.add_argument("--workers", default=4, type=int, help="number of data loading workers (default: 4)")

    parser.add_argument("--lr", default=0.05, type=float, help="initial learning rate (default: 0.01)")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum (default: 0.9)")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 5e-4)",
                        dest="weight_decay")
    parser.add_argument("--resume", default=False, type=bool, help="resume train from last epoch")
    return parser


def resume(model, optimizer, scaler, scheduler):
    if os.path.isfile(f"{name}" + '.pth'):
        loaded = torch.load(f"{name}" + '.pth')
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        scheduler.load_state_dict(loaded["scheduler"])
        start_epoch = loaded["epoch"]
        best_acc = loaded["acc"]
        print(f"loaded all parameters from best checkpoint, best_acc: {best_acc}.")
        print()
        return model, optimizer, scaler, scheduler, start_epoch, best_acc
    else:
        print(f"I can't find best.pth so we can't load params. ")
        return model, optimizer, scaler, scheduler, -1, 0.0


def train_one_epoch(model, loader, criterion, metric, optimizer, scheduler, scaler, epoch):
    model.train()
    loss_total = AverageMeter()
    metric_total = AverageMeter()
    semantic_total = AverageMeter()
    edge_total = AverageMeter()
    aux_total = AverageMeter()
    cutmix = CutMix(p=0.5)
    loop = tqdm.tqdm(loader, total=len(loader))
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs, targets = cutmix(inputs, targets)
        edge_targets = Target2Edge()(targets)
        inputs, targets, edge_targets = inputs.to(device), targets.to(device), edge_targets.to(device)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            semantic_loss, semantic_aux, edge_loss = criterion(outputs, targets, edge_targets)
            loss = (semantic_loss) + (0.4 * semantic_aux) + (edge_loss)
            acc = metric(outputs["out"].argmax(dim=1), targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_total.update(loss)
        metric_total.update(acc)
        semantic_total.update(semantic_loss)
        edge_total.update(edge_loss)
        aux_total.update(semantic_aux)
        torch.cuda.empty_cache()
        loop.set_description(f"Train -> Epoch:{epoch} Loss:{loss_total.avg:.4} Metric:{metric_total.avg:.4},"
                             f" Out Loss: {semantic_total.avg:.4} Aux Loss: {0.4 * aux_total.avg:.4},"
                             f" Edge Loss: {edge_total.avg:.4},"
                             f" LR: {get_lr(optimizer)}, Params: {count_parameters(model)}")

    writer.add_scalar('Loss/train', loss_total.avg.item(), epoch)
    writer.add_scalar('Metric/train', metric_total.avg.item(), epoch)
    writer.add_scalar('LR/train', get_lr(optimizer), epoch)
    state = {
        'model': model.state_dict(),
        'acc': metric_total.avg.item(),
        'epoch': epoch,
        'scaler': scaler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    path = f"{name}" + '.pth'
    torch.save(state, path)


def valid_one_epoch(model, loader, criterion, metric, optimizer, scheduler, scaler, epoch, ckpt):
    model.eval()
    loss_total = AverageMeter()
    metric_total = AverageMeter()
    semantic_total = AverageMeter()
    edge_total = AverageMeter()
    aux_total = AverageMeter()
    loop = tqdm.tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loop):
            edge_targets = Target2Edge()(targets)
            inputs, targets, edge_targets = inputs.to(device), targets.to(device), edge_targets.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs)
                semantic_loss, semantic_aux, edge_loss = criterion(outputs, targets, edge_targets)
                loss = (semantic_loss) + (0.4 * semantic_aux) + (edge_loss)
                acc = metric(outputs["out"].argmax(dim=1), targets)
            loss_total.update(loss)
            metric_total.update(acc)
            semantic_total.update(semantic_loss)
            edge_total.update(edge_loss)
            aux_total.update(semantic_aux)
            torch.cuda.empty_cache()
            loop.set_description(f"Valid -> Epoch:{epoch} Loss:{loss_total.avg:.4} Metric:{metric_total.avg:.4},"
                                 f" Out Loss: {semantic_total.avg:.4} Aux Loss: {0.4 * aux_total.avg:.4},"
                                 f" Edge Loss: {edge_total.avg:.4},"
                                 f" mIOU: {acc}")

    writer.add_scalar('Loss/valid', loss_total.avg.item(), epoch)
    writer.add_scalar('Metric/valid', metric_total.avg.item(), epoch)
    ckpt.save(model=model, acc=metric_total.avg.item(), filename=name, scaler=scaler, epoch=epoch,
              optimizer=optimizer, scheduler=scheduler)


def main():
    args = get_args().parse_args()
    num_classes = cityscapes_config["num_classes"]
    backbone = backbones[args.backbone]
    model = models[args.model](num_classes=num_classes, backbone=backbone)
    train_ds = CityScapes(phase="train", mean=model.backbone.mean, std=model.backbone.std)
    valid_ds = CityScapes(phase="val", mean=model.backbone.mean, std=model.backbone.std)
    # set_bn_momentum(model.backbone, momentum=0.01)

    configs = [["backbone", args.backbone],
               ["model", args.model],
               ["num classes", num_classes],
               ["num train data", len(train_ds)],
               ["num valid data", len(valid_ds)],
               ["batch size", args.batch_size],
               ["epochs", args.epochs],
               ["warmup iter", args.warmup_iter],
               ["workers", args.workers],
               ["lr", args.lr],
               ["momentum", args.momentum],
               ["weight decay", args.weight_decay],
               ["device", device],
               ["resume", args.resume], ]

    print(Fore.CYAN + tabulate(configs, headers=["name", "config"]))
    print()
    print(Fore.YELLOW + f"{name} summary")
    print(summary(model, (args.batch_size, 3, 512, 1024), device=device, col_width=16,
                  col_names=["output_size", "num_params", "mult_adds"], ))

    model = model.to(device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False,
                          num_workers=args.workers, drop_last=True)

    params = model.get_params(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_function(total_iterations=len(train_dl) * args.epochs,
                                                                       warmup_iters=args.warmup_iter,
                                                                       warmup_factor= 0.1,
                                                                       power=0.9))

    start_epoch, best_acc = 0, 0

    if args.resume:
        model, optimizer, scaler, scheduler, start_epoch, best_acc = resume(model, optimizer, scaler, scheduler)
        start_epoch += 1

    criterion = SegWithAuxLoss(ignore_index=cityscapes_config["ignore_label"], weight=train_ds.class_weights
                               ).to(device)

    metric = Metric(num_classes=num_classes)
    ckpt = Checkpoint(best_acc=best_acc)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model=model, loader=train_dl, criterion=criterion, metric=metric, optimizer=optimizer,
                        scaler=scaler, epoch=epoch, scheduler=scheduler)
        valid_one_epoch(model=model, loader=valid_dl, criterion=criterion, metric=metric, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, epoch=epoch, ckpt=ckpt)

        print()


if __name__ == "__main__":
    main()
