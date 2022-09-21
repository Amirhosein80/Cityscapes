import torch
import PIL
import numpy as np
import glob
import matplotlib.pyplot as plt
import albumentations as a
import os
import cv2
import torchvision.transforms.functional as f
import math

from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Rectangle
from models.edge import Target2Edge

cityscapes_config = {"train": ["data/cityscapes/leftImg8bit/train", "data/cityscapes/gtFine/train"],
                     "val": ["data/cityscapes/leftImg8bit/val", "data/cityscapes/gtFine/val"],
                     "test": ["data/cityscapes/leftImg8bit/test", "data/cityscapes/gtFine/test"],
                     "extra_train": ["data/cityscapes/leftImg8bit/train_extra", "data/cityscapes/gtCoarse/train_extra"],
                     "extra_val": ["data/cityscapes/leftImg8bit/val", "data/cityscapes/gtCoarse/val"],
                     "num_classes": 19, "ignore_label": 255, "base_size": (1024, 2048), "crop_size": (512, 512)}

cityscapes_classes = {0: {"name": "road", "color": (128, 64, 128)}, 1: {"name": "sidewalk", "color": (244, 35, 232)},
                      2: {"name": "building", "color": (70, 70, 70)}, 3: {"name": "wall", "color": (102, 102, 156)},
                      4: {"name": "fence", "color": (190, 153, 153)}, 5: {"name": "pole", "color": (153, 153, 153)},
                      6: {"name": "traffic light", "color": (250, 170, 30)},
                      7: {"name": "traffic sign", "color": (220, 220, 0)},
                      8: {"name": "vegetation", "color": (107, 142, 35)},
                      9: {"name": "terrain", "color": (152, 251, 152)},
                      10: {"name": "sky", "color": (70, 130, 180)}, 11: {"name": "person", "color": (220, 20, 60)},
                      12: {"name": "rider", "color": (255, 0, 0)}, 13: {"name": "car", "color": (0, 0, 142)},
                      14: {"name": "truck", "color": (0, 0, 70)}, 15: {"name": "bus", "color": (0, 60, 100)},
                      16: {"name": "train", "color": (0, 80, 100)},
                      17: {"name": "motorcycle", "color": (0, 0, 230)},
                      18: {"name": "bicycle", "color": (119, 11, 32)}, }

default_train_transform = a.Compose([
    a.RandomScale(scale_limit=(0.5, 2.0), p=0.8),
    a.RandomCrop(height=cityscapes_config["base_size"][0], width=cityscapes_config["base_size"][1]),
    a.Resize(height=cityscapes_config["crop_size"][0], width=cityscapes_config["crop_size"][1]),
    a.HorizontalFlip(p=0.5),
    a.Rotate(45, p=0.5),
    a.VerticalFlip(p=0.5),
    a.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5)

])
default_valid_transform = a.Compose([
])
default_test_transform = a.Compose([
])

default_extra_train_transform = a.Compose([
    a.Resize(height=416, width=416),
    a.RandomScale(scale_limit=(0.5, 2.0), p=1.0),
    a.RandomCrop(height=320, width=320),
    a.HorizontalFlip(p=0.5),
    a.VerticalFlip(p=0.5),
    a.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5)

])
default_extra_valid_transform = a.Compose([
    a.Resize(height=320, width=320)
])

class CityScapes(Dataset):
    def __init__(self, phase="train", transform=None, mean=None, std=None):
        self.images = glob.glob(cityscapes_config[phase][0] + "/*/*.png")
        self.phase = phase
        if mean is None:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        else:
            self.mean = torch.tensor(mean).reshape(3, 1, 1)
        if std is None:
            self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        else:
            self.std = torch.tensor(std).reshape(3, 1, 1)
        if transform is not None:
            self.transform = transform
        else:
            if phase == "train" :
                self.transform = default_train_transform
            elif phase == "val" :
                self.transform = default_valid_transform
            elif phase == "test":
                self.transform = default_test_transform
            elif phase == "extra_train":
                self.transform = default_extra_train_transform
            elif phase == "extra_val":
                self.transform = default_extra_valid_transform
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                                1.0166, 0.9969, 0.9754, 1.0489,
                                                0.8786, 1.0023, 0.9539, 0.9843,
                                                1.1116, 0.9037, 1.0865, 1.0955,
                                                1.0865, 1.1529, 1.0507])

        self.cityscapes_label_mapping = {-1: cityscapes_config["ignore_label"], 0: cityscapes_config["ignore_label"],
                                         1: cityscapes_config["ignore_label"], 2: cityscapes_config["ignore_label"],
                                         3: cityscapes_config["ignore_label"], 4: cityscapes_config["ignore_label"],
                                         5: cityscapes_config["ignore_label"], 6: cityscapes_config["ignore_label"],
                                         7: 0, 8: 1, 9: cityscapes_config["ignore_label"],
                                         10: cityscapes_config["ignore_label"], 11: 2, 12: 3,
                                         13: 4, 14: cityscapes_config["ignore_label"],
                                         15: cityscapes_config["ignore_label"],
                                         16: cityscapes_config["ignore_label"], 17: 5,
                                         18: cityscapes_config["ignore_label"],
                                         19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                                         25: 12, 26: 13, 27: 14, 28: 15,
                                         29: cityscapes_config["ignore_label"], 30: cityscapes_config["ignore_label"],
                                         31: 16, 32: 17, 33: 18}

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        image, mask = self.getitem(idx)
        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    def getitem(self, idx: int) -> (np.ndarray, np.ndarray):
        image = self.images[idx]
        image = image.replace("\\", "/")
        label = image.split("/")
        city, label = label[-2], label[-1]
        label = label.split("_leftImg8bit")[0]
        if self.phase == "extra_train" or self.phase == "extra_val":
            label = os.path.join(cityscapes_config[self.phase][1] + f"/{city}/", f"{label}_gtCoarse_labelIds.png")
        else:
            label = os.path.join(cityscapes_config[self.phase][1] + f"/{city}/", f"{label}_gtFine_labelIds.png")
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        label = self.convert_labels(label)
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']
        image = torch.tensor(image).permute(2, 0, 1) / 255.
        label = torch.tensor(label, dtype=torch.long)
        image = self.normalize(image)
        return image, label

    def show_image(self, idx: int) -> None:
        colors = []
        labels = []
        for v in cityscapes_classes.values():
            colors.append(list(v["color"]))
            labels.append(v["name"])
        colors = np.array(colors, dtype=np.uint8)
        handles = [Rectangle((0, 0), 1, 1, color=_c / 255) for _c in colors]
        image, mask = self.__getitem__(idx)
        image = self.denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = image * 255
        image = PIL.Image.fromarray(image.astype(np.uint8))
        mask = PIL.Image.fromarray(mask.numpy().astype(np.uint8))
        mask.putpalette(colors)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        plt.imshow(mask)
        plt.axis('off')
        plt.show()

        plt.imshow(np.ones((1, 1, 3)))
        plt.legend(handles, labels, loc="center")
        plt.axis('off')
        plt.show()

    def convert_labels(self, label: np.ndarray) -> np.ndarray:
        temp = label.copy()
        for k, v in self.cityscapes_label_mapping.items():
            label[temp == k] = v
        return label

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.sub_(self.mean).div_(self.std)
        return image

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.mul_(self.std).add_(self.mean)
        return image



class CutMix(torch.nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 1.0):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if torch.rand(1).item() >= self.p:
            return img, mask
        mask = mask.unsqueeze(1)
        img_rolled = img.roll(1, 0)
        mask_rolled = mask.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, _, H, W = img.shape
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        img[:, :, y1:y2, x1:x2] = img_rolled[:, :, y1:y2, x1:x2]
        mask[:, :, y1:y2, x1:x2] = mask_rolled[:, :, y1:y2, x1:x2]

        return img, mask.squeeze(1)


if __name__ == "__main__":
    ed = Target2Edge()
    ds = CityScapes("train")
    ds.show_image(12)
    image, mask = ds[10]
    # cu = CutMix(p=1.0)
    # dl = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0, drop_last=True, collate_fn=collater)
    # imgs, tar = next(iter(dl))
    # imgs, tar = cu(imgs, tar)
    # edge = ed(tar)
    # print(ds.denormalize(image).max())
    # print(torch.unique(mask))
