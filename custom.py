import torch
import torch.nn
from cityscapes import CityScapes, cityscapes_config, default_test_transform, cityscapes_classes
import numpy as np
from matplotlib.patches import Rectangle
import PIL
import matplotlib.pyplot as plt
from utils import Metric, AverageMeter
from models.deeplabv3plus import DeeplabV3Plus
from models.bisenetv2 import BisenetV2
from models.regseg import RegSeg
from models.repseg import RepSeg
from models.bisenet import Bisenet
from models.utils import MOBILENET_MEAN, MOBILENET_STD, IMAGENET_STD, IMAGENET_MEAN

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
metrics = Metric(num_classes=19)
colors = []
labels = []
for v in cityscapes_classes.values():
    colors.append(list(v["color"]))
    labels.append(v["name"])
colors = np.array(colors, dtype=np.uint8)
handles = [Rectangle((0, 0), 1, 1, color=_c / 255) for _c in colors]
bests = {}

dataset = CityScapes(phase="val", mean=IMAGENET_MEAN, std=IMAGENET_STD)
model = RepSeg(num_classes=cityscapes_config["num_classes"])
model.load_state_dict(torch.load("./best.pth")["model"])
model.eval()
model = model.to(device)
total_m = AverageMeter()
test_files = range(len(dataset))
for i in test_files:
    image, mask= dataset[i]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))["out"].argmax(dim=1)
    mi = metrics(output, mask.unsqueeze(0).to(device))
    if mi >= 0.89:
        bests.update({i : mi})
    total_m.update(mi)
    print(f"mIOU: {mi.item()}, step: {i}, avg: {total_m.avg.item()}")
    image = dataset.denormalize(image)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = image * 255
    image = PIL.Image.fromarray(image.astype(np.uint8))
    mask = PIL.Image.fromarray(mask.detach().cpu().numpy().astype(np.uint8))
    output = PIL.Image.fromarray(output[0].detach().cpu().numpy().astype(np.uint8))
    mask.putpalette(colors)
    output.putpalette(colors)
    m, o = np.array(mask), np.array(output)
    mo = np.hstack((m, o))
    mo = PIL.Image.fromarray(mo)
    mo.putpalette(colors)

    if mi.item() >= 0.86:
        mo.save(f"./outputs/outandtar_{i}.png")



