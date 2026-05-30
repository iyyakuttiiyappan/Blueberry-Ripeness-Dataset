from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _mask(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def _jitter(image: Image.Image, strength: float = 0.12) -> Image.Image:
    if strength <= 0:
        return image
    brightness = 1.0 + np.random.uniform(-strength, strength)
    contrast = 1.0 + np.random.uniform(-strength, strength)
    saturation = 1.0 + np.random.uniform(-strength, strength)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    return image


def _image_tensor(image: Image.Image, size: int, normalize: bool = True) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = F.to_tensor(image)
    if normalize:
        tensor = F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


class CropClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["crop_path"])
        if self.augment:
            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)
            image = _jitter(image)
        tensor = _image_tensor(image, self.image_size, normalize=True)
        return tensor, torch.tensor(int(row["label"]), dtype=torch.long), str(row["crop_path"])


class CountingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, target: str, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.target = target
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["image_path"])
        if self.augment:
            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)
            image = _jitter(image, strength=0.08)
        tensor = _image_tensor(image, self.image_size, normalize=True)
        if self.target == "classwise":
            class_cols = [col for col in self.frame.columns if col in row.index and not col.endswith("_components")]
            raise NotImplementedError(f"Use target='total' for now; classwise columns detected: {class_cols}")
        value = torch.tensor([float(row["Total"])], dtype=torch.float32)
        return tensor, value, str(row["image_path"])


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["image_path"])
        mask = _mask(row["semantic_mask_path"])
        if self.augment and np.random.rand() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.augment:
            image = _jitter(image, strength=0.08)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = F.normalize(F.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)
        return image_tensor, mask_tensor, str(row["image_path"])


class DetectionDataset(Dataset):
    def __init__(
        self,
        image_frame: pd.DataFrame,
        instances_frame: pd.DataFrame,
        image_size: int,
        include_masks: bool = False,
        augment: bool = False,
        mask_threshold: int = 127,
    ):
        self.image_frame = image_frame.reset_index(drop=True)
        self.instances_by_stem = {
            stem: group.reset_index(drop=True)
            for stem, group in instances_frame.groupby("stem", sort=False)
        }
        self.image_size = int(image_size)
        self.include_masks = bool(include_masks)
        self.augment = bool(augment)
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.image_frame)

    def _instance_masks(self, rows: pd.DataFrame, flip: bool) -> torch.Tensor:
        masks: list[torch.Tensor] = []
        labeled_cache: dict[str, np.ndarray] = {}
        for row in rows.itertuples(index=False):
            source_path = str(row.source_mask_path)
            if source_path not in labeled_cache:
                with Image.open(source_path) as image:
                    arr = np.asarray(image.convert("L")) > self.mask_threshold
                labeled_cache[source_path], _ = ndimage.label(arr)
            component = (labeled_cache[source_path] == int(row.component_number)).astype(np.uint8) * 255
            mask_image = Image.fromarray(component, mode="L")
            if flip:
                mask_image = ImageOps.mirror(mask_image)
            mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            masks.append(torch.as_tensor((np.asarray(mask_image).copy() > 0), dtype=torch.uint8))
        if not masks:
            return torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
        return torch.stack(masks, dim=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.image_frame.iloc[index]
        image = _rgb(row["image_path"])
        original_width = int(row["aligned_width"])
        original_height = int(row["aligned_height"])
        rows = self.instances_by_stem.get(row["stem"], pd.DataFrame())
        flip = bool(self.augment and np.random.rand() < 0.5)
        if flip:
            image = ImageOps.mirror(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_tensor = F.to_tensor(image)

        if rows.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            boxes_np = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).copy()
            if flip:
                x1 = boxes_np[:, 0].copy()
                x2 = boxes_np[:, 2].copy()
                boxes_np[:, 0] = original_width - x2
                boxes_np[:, 2] = original_width - x1
            boxes_np[:, [0, 2]] *= self.image_size / original_width
            boxes_np[:, [1, 3]] *= self.image_size / original_height
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(rows["det_label"].to_numpy(dtype=np.int64).copy(), dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            masks = self._instance_masks(rows, flip=flip) if self.include_masks else None

        target: dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.include_masks:
            target["masks"] = masks
        return image_tensor, target


def detection_collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)
