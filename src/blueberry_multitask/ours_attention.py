from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import roi_align
from torchvision.transforms import functional as VF
from tqdm import tqdm

from .annotations import prepare_annotations
from .config import output_dirs
from .datasets import IMAGENET_MEAN, IMAGENET_STD, _jitter, _mask, _rgb
from .metrics import (
    SegmentationMeter,
    classification_metrics,
    classification_report_df,
    confusion_df,
    counting_metrics,
    detection_metrics,
)
from .ours import (
    _class_weights,
    _components_from_prediction,
    _dice_loss,
    _gt_by_image,
    _loader_kwargs,
    _mirror_task_outputs,
    _split,
)
from .plots import save_confusion_matrix, save_history_plot, save_prediction_overlay
from .utils import json_dump, now_stamp, resolve_device, set_seed


ROI_METHOD = "berrymtl_roi_attention"
ROI_DISPLAY = "BerryMTL-ROI-Attention (ours)"


class BerryMTLInstanceDataset(Dataset):
    def __init__(
        self,
        image_frame: pd.DataFrame,
        instances: pd.DataFrame,
        image_size: int,
        augment: bool,
        tile_prob: float = 0.0,
        tile_min_scale: float = 0.42,
        tile_max_scale: float = 0.72,
        tile_min_visibility: float = 0.25,
        tile_anchor_class_power: float = 0.0,
    ):
        self.image_frame = image_frame.reset_index(drop=True)
        self.instances_by_stem = {
            stem: group.reset_index(drop=True)
            for stem, group in instances.groupby("stem", sort=False)
        }
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.tile_prob = float(tile_prob)
        self.tile_min_scale = float(tile_min_scale)
        self.tile_max_scale = float(tile_max_scale)
        self.tile_min_visibility = float(tile_min_visibility)
        self.tile_anchor_class_power = float(tile_anchor_class_power)
        if self.tile_anchor_class_power > 0 and "class_index" in instances:
            counts = instances["class_index"].value_counts().to_dict()
            max_count = max([float(value) for value in counts.values()] or [1.0])
            self.tile_anchor_class_weights = {
                int(label): float(np.clip((max_count / max(float(count), 1.0)) ** self.tile_anchor_class_power, 1.0, 12.0))
                for label, count in counts.items()
            }
        else:
            self.tile_anchor_class_weights = {}

    def __len__(self) -> int:
        return len(self.image_frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor, torch.Tensor]:
        row = self.image_frame.iloc[index]
        image = _rgb(row["image_path"])
        mask = _mask(row["semantic_mask_path"])
        original_width = float(row["aligned_width"])
        original_height = float(row["aligned_height"])
        rows = self.instances_by_stem.get(row["stem"], pd.DataFrame())
        boxes_override: np.ndarray | None = None
        tile_count: float | None = None

        if self.augment and self.tile_prob > 0.0 and not rows.empty and np.random.rand() < self.tile_prob:
            full_boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).copy()
            centers_x = (full_boxes[:, 0] + full_boxes[:, 2]) * 0.5
            centers_y = (full_boxes[:, 1] + full_boxes[:, 3]) * 0.5
            areas = np.maximum(1.0, (full_boxes[:, 2] - full_boxes[:, 0]) * (full_boxes[:, 3] - full_boxes[:, 1]))
            min_side = max(2.0, min(original_width, original_height))
            low = max(0.10, min(self.tile_min_scale, self.tile_max_scale))
            high = min(1.0, max(self.tile_min_scale, self.tile_max_scale))
            for _ in range(8):
                tile_side = float(np.random.uniform(low, high) * min_side)
                tile_side = float(max(2, min(int(round(tile_side)), int(original_width), int(original_height))))
                if self.tile_anchor_class_weights:
                    labels = rows["class_index"].to_numpy(dtype=np.int64, copy=True)
                    weights = np.asarray([self.tile_anchor_class_weights.get(int(label), 1.0) for label in labels], dtype=np.float64)
                    weights = weights / max(float(weights.sum()), 1e-9)
                    anchor_idx = int(np.random.choice(np.arange(len(full_boxes)), p=weights))
                else:
                    anchor_idx = int(np.random.randint(0, len(full_boxes)))
                crop_x1 = float(np.clip(centers_x[anchor_idx] - np.random.uniform(0.30, 0.70) * tile_side, 0.0, original_width - tile_side))
                crop_y1 = float(np.clip(centers_y[anchor_idx] - np.random.uniform(0.30, 0.70) * tile_side, 0.0, original_height - tile_side))
                crop_x1 = float(int(np.clip(int(round(crop_x1)), 0, int(original_width - tile_side))))
                crop_y1 = float(int(np.clip(int(round(crop_y1)), 0, int(original_height - tile_side))))
                crop_x2 = crop_x1 + tile_side
                crop_y2 = crop_y1 + tile_side
                clipped = full_boxes.copy()
                clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], crop_x1, crop_x2) - crop_x1
                clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], crop_y1, crop_y2) - crop_y1
                visible_area = np.maximum(0.0, clipped[:, 2] - clipped[:, 0]) * np.maximum(0.0, clipped[:, 3] - clipped[:, 1])
                centers_inside = (centers_x >= crop_x1) & (centers_x <= crop_x2) & (centers_y >= crop_y1) & (centers_y <= crop_y2)
                keep = centers_inside & ((visible_area / areas) >= self.tile_min_visibility)
                keep &= (clipped[:, 2] - clipped[:, 0] >= 2.0) & (clipped[:, 3] - clipped[:, 1] >= 2.0)
                if not bool(np.any(keep)):
                    continue
                crop_box = (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))
                image = image.crop(crop_box)
                mask = mask.crop(crop_box)
                rows = rows.iloc[np.where(keep)[0]].reset_index(drop=True)
                boxes_override = clipped[keep].astype(np.float32, copy=False)
                original_width = float(crop_box[2] - crop_box[0])
                original_height = float(crop_box[3] - crop_box[1])
                tile_count = float(len(rows))
                break

        flip = bool(self.augment and np.random.rand() < 0.5)
        if flip:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.augment:
            image = _jitter(image, strength=0.08)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = VF.normalize(VF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)
        count = torch.tensor([float(row["Total"]) if tile_count is None else tile_count], dtype=torch.float32)

        if rows.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_np = boxes_override.copy() if boxes_override is not None else rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).copy()
            if flip:
                x1 = boxes_np[:, 0].copy()
                x2 = boxes_np[:, 2].copy()
                boxes_np[:, 0] = original_width - x2
                boxes_np[:, 2] = original_width - x1
            boxes_np[:, [0, 2]] *= self.image_size / original_width
            boxes_np[:, [1, 3]] *= self.image_size / original_height
            boxes_np[:, 0::2] = np.clip(boxes_np[:, 0::2], 0, self.image_size)
            boxes_np[:, 1::2] = np.clip(boxes_np[:, 1::2], 0, self.image_size)
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(rows["class_index"].to_numpy(dtype=np.int64).copy(), dtype=torch.long)
        return image_tensor, mask_tensor, count, str(row["stem"]), boxes, labels


def instance_collate(batch):
    images, masks, counts, stems, boxes, labels = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(counts, dim=0),
        list(stems),
        list(boxes),
        list(labels),
    )


class BerryMTLROIAttentionNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        decoder_channels: int = 128,
        roi_channels: int = 192,
        roi_size: int = 7,
    ):
        super().__init__()
        import timm

        self.num_classes = int(num_classes)
        self.roi_size = int(roi_size)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels])
        decoder_in = decoder_channels * len(channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_in, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )
        self.roi_feature = nn.Sequential(
            nn.Conv2d(decoder_in, roi_channels, kernel_size=1),
            nn.BatchNorm2d(roi_channels),
            nn.GELU(),
        )
        self.roi_attention = nn.Sequential(
            nn.Conv2d(roi_channels + 1, roi_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(roi_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(roi_channels),
            nn.Linear(roi_channels, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes - 1),
        )
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim == 4 and feature.shape[1] != expected_channels and feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    @staticmethod
    def _rois_from_boxes(boxes: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        rows = []
        for batch_idx, batch_boxes in enumerate(boxes):
            if batch_boxes.numel() == 0:
                continue
            batch_boxes = batch_boxes.to(device=device, dtype=torch.float32)
            indices = torch.full((batch_boxes.shape[0], 1), float(batch_idx), device=device, dtype=torch.float32)
            rows.append(torch.cat([indices, batch_boxes], dim=1))
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32, device=device)
        return torch.cat(rows, dim=0)

    def classify_rois(self, roi_feature: torch.Tensor, seg_logits: torch.Tensor, boxes: list[torch.Tensor], image_size: int) -> torch.Tensor:
        rois = self._rois_from_boxes(boxes, roi_feature.device)
        if rois.numel() == 0:
            return torch.zeros((0, self.num_classes - 1), dtype=roi_feature.dtype, device=roi_feature.device)
        spatial_scale = roi_feature.shape[-1] / float(image_size)
        pooled = roi_align(
            roi_feature,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )
        foreground = torch.softmax(seg_logits, dim=1)[:, 1:].sum(dim=1, keepdim=True)
        pooled_fg = roi_align(
            foreground,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True,
        )
        attn = self.roi_attention(torch.cat([pooled, pooled_fg], dim=1))
        gated = pooled * (0.5 + attn)
        return self.class_head(gated)

    def forward(self, x: torch.Tensor, boxes: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        channels = list(self.encoder.feature_info.channels())
        features = [self._nchw(feature, channel) for feature, channel in zip(features, channels)]
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(conv(feature), size=target_size, mode="bilinear", align_corners=False)
            for feature, conv in zip(features, self.lateral)
        ]
        decoder = torch.cat(projected, dim=1)
        logits = self.seg_head(decoder)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        roi_feature = self.roi_feature(decoder)
        output = {
            "seg": logits,
            "count": self.count_head(features[-1]),
            "roi_feature": roi_feature,
        }
        if boxes is not None:
            output["cls"] = self.classify_rois(roi_feature, logits, boxes, image_size=int(input_size[-1]))
        return output


def _instance_class_weights(instances: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = instances["class_index"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy(dtype=np.float64)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / max(float(weights.mean()), 1e-9)
    weights = np.clip(weights, 0.25, 8.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def _flatten_labels(labels: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    valid = [value.to(device=device, dtype=torch.long) for value in labels if value.numel() > 0]
    if not valid:
        return torch.zeros((0,), dtype=torch.long, device=device)
    return torch.cat(valid, dim=0)


def _roi_epoch(
    model: nn.Module,
    loader: DataLoader,
    seg_criterion: nn.Module,
    cls_criterion: nn.Module,
    count_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    train: bool,
    num_classes: int,
    class_names: list[str],
    count_weight: float,
    dice_weight: float,
    cls_weight: float,
    cls_score_weight: float,
    count_score_weight: float,
) -> dict[str, float]:
    model.train(train)
    meter = SegmentationMeter(num_classes)
    count_true: list[float] = []
    count_pred: list[float] = []
    cls_true: list[int] = []
    cls_pred: list[int] = []
    total_loss = 0.0
    seg_loss_total = 0.0
    count_loss_total = 0.0
    cls_loss_total = 0.0
    image_count = 0
    roi_count = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")

    for images, masks, targets, _, boxes, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        boxes = [value.to(device, non_blocking=True) for value in boxes]
        cls_labels = _flatten_labels(labels, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images, boxes=boxes)
            logits = output["seg"]
            counts = output["count"].float()
            seg_loss = seg_criterion(logits, masks) + dice_weight * _dice_loss(logits, masks, num_classes)
            c_loss = count_loss(counts, targets.float())
            if cls_labels.numel() > 0:
                cls_logits = output["cls"]
                r_loss = cls_criterion(cls_logits, cls_labels)
            else:
                cls_logits = torch.zeros((0, num_classes - 1), device=device)
                r_loss = torch.zeros((), device=device)
            loss = seg_loss + count_weight * c_loss + cls_weight * r_loss
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch = images.shape[0]
        image_count += batch
        total_loss += float(loss.item()) * batch
        seg_loss_total += float(seg_loss.item()) * batch
        count_loss_total += float(c_loss.item()) * batch
        cls_loss_total += float(r_loss.item()) * max(1, int(cls_labels.numel()))
        roi_count += int(cls_labels.numel())
        meter.update(logits.detach(), masks.detach())
        count_true.extend(targets.detach().cpu().numpy().reshape(-1).tolist())
        count_pred.extend(counts.detach().cpu().numpy().reshape(-1).tolist())
        if cls_labels.numel() > 0:
            cls_true.extend(cls_labels.detach().cpu().tolist())
            cls_pred.extend(cls_logits.detach().argmax(dim=1).cpu().tolist())

    seg_metrics, _ = meter.metrics(class_names)
    c_metrics = counting_metrics(np.asarray(count_true), np.asarray(count_pred))
    cls_metrics = classification_metrics(cls_true, cls_pred) if cls_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "balanced_accuracy": 0.0}
    output = {
        "loss": total_loss / max(1, image_count),
        "seg_loss": seg_loss_total / max(1, image_count),
        "count_loss": count_loss_total / max(1, image_count),
        "roi_cls_loss": cls_loss_total / max(1, roi_count),
        **{f"seg_{key}": value for key, value in seg_metrics.items()},
        **{f"count_{key}": value for key, value in c_metrics.items()},
        **{f"cls_{key}": value for key, value in cls_metrics.items()},
    }
    output["joint_score"] = float(
        seg_metrics["miou_foreground"] + cls_score_weight * cls_metrics["macro_f1"] - count_score_weight * c_metrics["mae"]
    )
    return output


def _classify_gt_rois(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    stem: str,
    filename: str,
    path: str,
    start_index: int,
) -> pd.DataFrame:
    if logits.numel() == 0:
        return pd.DataFrame()
    probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    y_true = labels.detach().cpu().numpy().astype(int)
    rows = []
    for offset, (true_value, pred_value, confidence) in enumerate(zip(y_true, pred, conf)):
        rows.append(
            {
                "instance_index": start_index + offset,
                "y_true": int(true_value),
                "y_pred": int(pred_value),
                "confidence": float(confidence),
                "true_class": class_names[int(true_value)],
                "pred_class": class_names[int(pred_value)],
                "stem": stem,
                "filename": filename,
                "path": path,
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def _evaluate_roi(
    model: BerryMTLROIAttentionNet,
    loader: DataLoader,
    image_frame: pd.DataFrame,
    instances: pd.DataFrame,
    device: torch.device,
    amp: bool,
    image_size: int,
    class_names: list[str],
    min_component_area: int,
    classify_detection_boxes: bool,
) -> dict[str, Any]:
    num_classes = len(class_names) + 1
    model.eval()
    image_by_stem = image_frame.set_index("stem")
    instance_targets = _gt_by_image(image_frame, instances, image_size)
    seg_meter = SegmentationMeter(num_classes)
    count_true: list[float] = []
    count_pred: list[float] = []
    detection_predictions: list[dict[str, np.ndarray]] = []
    detection_targets: list[dict[str, np.ndarray]] = []
    detection_rows: list[dict[str, Any]] = []
    classification_rows: list[pd.DataFrame] = []
    segmentation_rows: list[dict[str, Any]] = []
    sample_payload = None

    stem_order = image_frame["stem"].astype(str).tolist()
    image_index = {stem: idx for idx, stem in enumerate(stem_order)}
    for images, masks, targets, stems, boxes, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boxes = [value.to(device, non_blocking=True) for value in boxes]
        cls_labels = _flatten_labels(labels, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images, boxes=boxes)
            logits = output["seg"]
            counts = output["count"].float()
            cls_logits = output["cls"]
        probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_batch = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        true_batch = masks.detach().cpu().numpy().astype(np.uint8)
        seg_meter.update(logits, masks)
        count_true.extend(targets.numpy().reshape(-1).tolist())
        count_pred.extend(counts.detach().cpu().numpy().reshape(-1).tolist())

        cls_offset = 0
        det_boxes_for_batch: list[torch.Tensor] = []
        det_by_stem: dict[str, dict[str, np.ndarray]] = {}
        for local_idx, stem in enumerate(stems):
            stem = str(stem)
            row = image_by_stem.loc[stem]
            true_mask = true_batch[local_idx]
            pred_mask = pred_batch[local_idx]
            true_fg = true_mask > 0
            pred_fg = pred_mask > 0
            segmentation_rows.append(
                {
                    "stem": stem,
                    "filename": row["filename"],
                    "foreground_iou": float((true_fg & pred_fg).sum() / max(1, (true_fg | pred_fg).sum())),
                    "foreground_pixel_error": int(pred_fg.sum() - true_fg.sum()),
                }
            )
            pred_det = _components_from_prediction(pred_mask, probs_batch[local_idx], min_component_area)
            det_by_stem[stem] = pred_det
            det_boxes_for_batch.append(torch.as_tensor(pred_det["boxes"], dtype=torch.float32, device=device))
            target_det = instance_targets[stem]
            detection_targets.append(target_det)

            roi_count = int(labels[local_idx].numel())
            if roi_count > 0:
                frame_logits = cls_logits[cls_offset : cls_offset + roi_count]
                frame_labels = cls_labels[cls_offset : cls_offset + roi_count]
                classification_rows.append(
                    _classify_gt_rois(
                        frame_logits,
                        frame_labels,
                        class_names,
                        stem,
                        str(row["filename"]),
                        str(row["image_path"]),
                        start_index=0,
                    )
                )
            cls_offset += roi_count
            if sample_payload is None:
                sample_payload = (row["image_path"], true_mask, pred_mask)

        if classify_detection_boxes and any(value.numel() > 0 for value in det_boxes_for_batch):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                det_logits = model.classify_rois(output["roi_feature"], logits, det_boxes_for_batch, image_size=image_size)
            det_probs = torch.softmax(det_logits.detach().float(), dim=1).cpu().numpy() if det_logits.numel() else np.zeros((0, len(class_names)))
        else:
            det_probs = np.zeros((0, len(class_names)))

        det_offset = 0
        for local_idx, stem in enumerate(stems):
            stem = str(stem)
            pred_det = det_by_stem[stem]
            if classify_detection_boxes and len(pred_det["boxes"]) > 0 and det_probs.size:
                local_probs = det_probs[det_offset : det_offset + len(pred_det["boxes"])]
                det_offset += len(pred_det["boxes"])
                class_pred = local_probs.argmax(axis=1) + 1
                class_conf = local_probs.max(axis=1)
                pred_det = {
                    "boxes": pred_det["boxes"],
                    "labels": class_pred.astype(int),
                    "scores": (pred_det["scores"] * class_conf).astype(float),
                }
            detection_predictions.append(pred_det)
            for box, label, score in zip(pred_det["boxes"], pred_det["labels"], pred_det["scores"]):
                detection_rows.append(
                    {
                        "batch_image_index": image_index[stem],
                        "label": int(label),
                        "score": float(score),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    }
                )

    seg_metrics, per_class_seg = seg_meter.metrics(class_names)
    count_metrics = counting_metrics(np.asarray(count_true), np.asarray(count_pred))
    detection_metric_values, per_class_det = detection_metrics(detection_predictions, detection_targets, len(class_names))
    cls_predictions = pd.concat([frame for frame in classification_rows if not frame.empty], ignore_index=True) if classification_rows else pd.DataFrame()
    cls_metrics = classification_metrics(cls_predictions["y_true"].tolist(), cls_predictions["y_pred"].tolist())
    count_predictions = pd.DataFrame(
        {
            "path": image_frame["image_path"].tolist(),
            "stem": image_frame["stem"].tolist(),
            "y_true": count_true,
            "y_pred": count_pred,
            "error": np.asarray(count_pred) - np.asarray(count_true),
        }
    )
    return {
        "segmentation": seg_metrics,
        "segmentation_per_class": per_class_seg,
        "segmentation_per_image": pd.DataFrame(segmentation_rows),
        "counting": count_metrics,
        "counting_predictions": count_predictions,
        "detection": detection_metric_values,
        "detection_per_class": per_class_det,
        "detection_predictions": pd.DataFrame(detection_rows),
        "classification": cls_metrics,
        "classification_predictions": cls_predictions,
        "classification_report": classification_report_df(cls_predictions["y_true"].tolist(), cls_predictions["y_pred"].tolist(), class_names),
        "classification_confusion": confusion_df(cls_predictions["y_true"].tolist(), cls_predictions["y_pred"].tolist(), class_names),
        "sample": sample_payload,
    }


def _metadata(
    task: str,
    seed: int,
    device: torch.device,
    image_size: int,
    model_name: str,
    parameters: int,
    method: str,
    display_name: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "method": method,
        "display_name": display_name,
        "family": "Unified multitask ROI attention",
        "resolved_model_name": f"timm_fpn_roi_attention:{model_name}",
        "seed": seed,
        "device": str(device),
        "parameters": int(parameters),
        "trainable_parameters": int(parameters),
        "image_size": int(image_size),
        "shared_backbone": True,
        "roi_attention": True,
        "classification_head": "mask-attended ROI head",
    }


def run_berrymtl_roi_attention(
    config: dict[str, Any],
    seed: int | None = None,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
    profile_key: str = "ours_roi_attention",
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    seed = int(seed if seed is not None else config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(device_name)
    classes = list(config["classes"])
    num_classes = len(classes) + 1
    cfg = {**config.get("ours", {}), **config.get(profile_key, {})}
    method = str(cfg.get("method", ROI_METHOD))
    display_name = str(cfg.get("display_name", ROI_DISPLAY))
    model_name = str(cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))
    image_size = int(cfg.get("image_size", 512))
    batch_size = int(batch_size if batch_size is not None else cfg.get("batch_size", 3))
    epochs = int(epochs if epochs is not None else cfg.get("epochs", 60))
    count_weight = float(cfg.get("count_loss_weight", 0.035))
    dice_weight = float(cfg.get("dice_loss_weight", 0.6))
    cls_weight = float(cfg.get("classification_loss_weight", 0.55))
    cls_score_weight = float(cfg.get("classification_score_weight", 0.25))
    count_score_weight = float(cfg.get("count_score_weight", 0.002))
    min_component_area = int(cfg.get("min_component_area", 8))
    classify_detection_boxes = bool(cfg.get("classify_detection_boxes", True))
    pretrained = bool(config.get("training", {}).get("pretrained", True) if pretrained is None else pretrained)

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    train_images = _split(images, "train", limit)
    val_images = _split(images, "val", limit)
    test_images = _split(images, "test", limit)
    train_instances = instances[instances["stem"].isin(set(train_images["stem"]))]
    val_instances = instances[instances["stem"].isin(set(val_images["stem"]))]
    test_instances = instances[instances["stem"].isin(set(test_images["stem"]))]
    loader_kwargs = _loader_kwargs(config, device)
    train_loader = DataLoader(
        BerryMTLInstanceDataset(train_images, train_instances, image_size, True),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=instance_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        BerryMTLInstanceDataset(val_images, val_instances, image_size, False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=instance_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        BerryMTLInstanceDataset(test_images, test_instances, image_size, False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=instance_collate,
        **loader_kwargs,
    )

    model = BerryMTLROIAttentionNet(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        decoder_channels=int(cfg.get("decoder_channels", 128)),
        roi_channels=int(cfg.get("roi_channels", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
    ).to(device)
    init_source = None
    if bool(cfg.get("init_from_unified", False)):
        source_root = dirs["analysis"] / "ours" / "runs"
        candidates = sorted(source_root.glob("*_berrymtl_unified_seed*"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No BerryMTL unified checkpoint found under {source_root}")
        init_source = candidates[0] / "best.pt"
        checkpoint = torch.load(init_source, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
    if bool(cfg.get("train_roi_only", False)):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in [model.roi_feature, model.roi_attention, model.class_head]:
            for parameter in module.parameters():
                parameter.requires_grad = True
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("lr", config.get("training", {}).get("lr", 0.0003))),
        weight_decay=float(cfg.get("weight_decay", config.get("training", {}).get("weight_decay", 0.05))),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    seg_weights = _class_weights(train_images, num_classes).to(device)
    cls_weights = _instance_class_weights(train_instances, len(classes)).to(device)
    seg_loss = nn.CrossEntropyLoss(weight=seg_weights)
    cls_loss = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=float(cfg.get("label_smoothing", 0.03)))
    count_loss = nn.SmoothL1Loss()
    amp = bool(config.get("training", {}).get("amp", True))
    patience = int(cfg.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 8)))

    run_dir = dirs["analysis"] / "ours" / "runs" / f"{now_stamp()}_{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _roi_epoch(
            model,
            train_loader,
            seg_loss,
            cls_loss,
            count_loss,
            optimizer,
            device,
            amp,
            True,
            num_classes,
            classes,
            count_weight,
            dice_weight,
            cls_weight,
            cls_score_weight,
            count_score_weight,
        )
        val_metrics = _roi_epoch(
            model,
            val_loader,
            seg_loss,
            cls_loss,
            count_loss,
            optimizer,
            device,
            amp,
            False,
            num_classes,
            classes,
            count_weight,
            dice_weight,
            cls_weight,
            cls_score_weight,
            count_score_weight,
        )
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        save_history_plot(history_df.rename(columns={"val_joint_score": "val_joint"}), run_dir / "training_curves.png", "joint")
        score = float(val_metrics["joint_score"])
        if best_value is None or score > best_value:
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = _evaluate_roi(
        model,
        test_loader,
        test_images,
        test_instances,
        device,
        amp,
        image_size,
        classes,
        min_component_area=min_component_area,
        classify_detection_boxes=classify_detection_boxes,
    )
    train_seconds = time.perf_counter() - start

    test["segmentation_per_class"].to_csv(run_dir / "segmentation_per_class_test.csv", index=False)
    test["segmentation_per_image"].to_csv(run_dir / "segmentation_per_image_test.csv", index=False)
    test["counting_predictions"].to_csv(run_dir / "counting_predictions_test.csv", index=False)
    test["detection_per_class"].to_csv(run_dir / "detection_per_class_test.csv", index=False)
    test["detection_predictions"].to_csv(run_dir / "detection_predictions_test.csv", index=False)
    test["classification_predictions"].to_csv(run_dir / "classification_predictions_test.csv", index=False)
    test["classification_report"].to_csv(run_dir / "classification_report_test.csv", index=False)
    test["classification_confusion"].to_csv(run_dir / "confusion_matrix_test.csv")
    save_confusion_matrix(test["classification_confusion"], run_dir / "confusion_matrix_test.png", title=f"{display_name}: ROI Classification")
    if test["sample"] is not None:
        image_path, target, pred = test["sample"]
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, target, pred, run_dir / "sample_prediction_overlay.jpg")

    common = {"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": train_seconds}
    json_dump(
        {
            "method": method,
            "display_name": display_name,
            "best_epoch": best_epoch,
            "best_val_metric": best_value,
            "classification_loss_weight": cls_weight,
            "classification_score_weight": cls_score_weight,
            "classify_detection_boxes": classify_detection_boxes,
            "init_source": str(init_source.resolve()) if init_source is not None else None,
            "train_roi_only": bool(cfg.get("train_roi_only", False)),
        },
        run_dir / "metadata.json",
    )
    combined = {
        "classification": {**test["classification"], **common},
        "counting": {**test["counting"], **common},
        "segmentation": {**test["segmentation"], **common},
        "detection": {**test["detection"], **common},
    }
    json_dump(combined, run_dir / "test_metrics_by_task.json")

    base_meta = _metadata("multitask", seed, device, image_size, model_name, parameters, method, display_name)
    base_meta["trainable_parameters"] = int(trainable_parameters)
    if init_source is not None:
        base_meta["init_source"] = str(init_source.resolve())
    base_meta["train_roi_only"] = bool(cfg.get("train_roi_only", False))
    _mirror_task_outputs(
        config,
        run_dir,
        "classification",
        {**test["classification"], **common},
        {**base_meta, "task": "classification"},
        {
            "predictions_test.csv": run_dir / "classification_predictions_test.csv",
            "classification_report_test.csv": run_dir / "classification_report_test.csv",
            "confusion_matrix_test.csv": run_dir / "confusion_matrix_test.csv",
            "confusion_matrix_test.png": run_dir / "confusion_matrix_test.png",
        },
    )
    _mirror_task_outputs(
        config,
        run_dir,
        "counting",
        {**test["counting"], **common},
        {**base_meta, "task": "counting"},
        {"predictions_test.csv": run_dir / "counting_predictions_test.csv"},
    )
    _mirror_task_outputs(
        config,
        run_dir,
        "segmentation",
        {**test["segmentation"], **common},
        {**base_meta, "task": "segmentation"},
        {
            "per_class_test.csv": run_dir / "segmentation_per_class_test.csv",
            "per_image_test.csv": run_dir / "segmentation_per_image_test.csv",
            "sample_prediction_overlay.jpg": run_dir / "sample_prediction_overlay.jpg",
        },
    )
    det_per_class = test["detection_per_class"].copy()
    det_per_class["class_name"] = det_per_class["class_id"].map({idx + 1: name for idx, name in enumerate(classes)})
    det_per_class.to_csv(run_dir / "detection_per_class_test_named.csv", index=False)
    _mirror_task_outputs(
        config,
        run_dir,
        "detection",
        {**test["detection"], **common},
        {**base_meta, "task": "detection"},
        {
            "predictions_test.csv": run_dir / "detection_predictions_test.csv",
            "per_class_test.csv": run_dir / "detection_per_class_test_named.csv",
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir
