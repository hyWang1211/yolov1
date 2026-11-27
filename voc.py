from dataclasses import dataclass
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from yolo import CLASS_COUNT, GRID_SIZE, IMAGE_SIZE, VOC_CLASSES

CLASS_IDS = {name: idx for idx, name in enumerate(VOC_CLASSES)}
DEFAULT_VOC_ROOT = Path("dataset/VOC2012_train_val")


@dataclass
class VocObjects:
    boxes: torch.Tensor
    labels: torch.Tensor
    difficult: torch.Tensor


def parse_objects(xml_path, image_size):
    width, height = image_size
    boxes, labels, difficult = [], [], []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        bounds = obj.find("bndbox")
        # VOC uses one-based inclusive corners. Use zero-based half-open image edges.
        left = float(bounds.findtext("xmin")) - 1
        top = float(bounds.findtext("ymin")) - 1
        right = float(bounds.findtext("xmax"))
        bottom = float(bounds.findtext("ymax"))
        boxes.append((left / width, top / height, right / width, bottom / height))
        labels.append(CLASS_IDS[obj.findtext("name")])
        difficult.append(obj.findtext("difficult", "0") == "1")
    return VocObjects(
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp_(0, 1),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(difficult, dtype=torch.bool),
    )


def encode_target(objects):
    """Encode one object per cell as (x, y, w, h, objectness, one-hot class)."""
    target = torch.zeros(GRID_SIZE, GRID_SIZE, 5 + CLASS_COUNT)
    sizes = objects.boxes[:, 2:] - objects.boxes[:, :2]
    areas = sizes.prod(dim=1)
    # Keep the largest eligible object when centers share a cell.
    valid = (sizes > 0).all(dim=1) & ~objects.difficult
    indices = torch.where(valid)[0]
    indices = indices[areas[indices].argsort(descending=True, stable=True)]
    for index in indices.tolist():
        box = objects.boxes[index]
        center = (box[:2] + box[2:]) * (0.5 * GRID_SIZE)
        col, row = center.long().clamp(max=GRID_SIZE - 1).tolist()
        cell = target[row, col]
        if cell[4]:
            continue
        cell[:2] = center - center.new_tensor((col, row))
        cell[2:4] = sizes[index]
        cell[4] = 1
        cell[5 + objects.labels[index]] = 1
    return target


def image_tensor(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(image, dtype=np.uint8))
    return pixels.permute(2, 0, 1).contiguous().float().div_(255)


def augment(image, boxes):
    left, top = (random.uniform(-0.2, 0.2) for _ in range(2))
    right, bottom = (1 + random.uniform(-0.2, 0.2) for _ in range(2))
    width, height = image.size
    image = image.transform(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Transform.EXTENT,
        (left * width, top * height, right * width, bottom * height),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(114, 114, 114),
    )
    offsets = boxes.new_tensor((left, top, left, top))
    scales = boxes.new_tensor((right - left, bottom - top) * 2)
    boxes = ((boxes - offsets) / scales).clamp_(0, 1)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]

    hue, saturation, value = image.convert("HSV").split()
    saturation_scale = random.uniform(1 / 1.5, 1.5)
    value_scale = random.uniform(1 / 1.5, 1.5)
    saturation = saturation.point([min(round(i * saturation_scale), 255) for i in range(256)])
    value = value.point([min(round(i * value_scale), 255) for i in range(256)])
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB"), boxes


class VocDetection(Dataset):
    def __init__(self, root, split, augment_images=False):
        self.root = Path(root)
        split_path = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = split_path.read_text(encoding="utf-8").split()
        self.augment_images = augment_images

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        with Image.open(self.root / "JPEGImages" / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
        objects = parse_objects(self.root / "Annotations" / f"{image_id}.xml", image.size)
        if self.augment_images:
            image, objects.boxes = augment(image, objects.boxes)
        return image_tensor(image), encode_target(objects), objects


def collate_detections(batch):
    images, targets, objects = zip(*batch)
    return torch.stack(images), torch.stack(targets), objects
