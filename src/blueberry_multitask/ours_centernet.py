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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import nms, roi_align
from tqdm import tqdm

from .annotations import prepare_annotations
from .config import output_dirs
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
    _dice_loss,
    _gt_by_image,
    _loader_kwargs,
    _mirror_task_outputs,
    _split,
)
from .ours_attention import (
    BerryMTLInstanceDataset,
    _classify_gt_rois,
    _flatten_labels,
    _instance_class_weights,
    instance_collate,
)
from .plots import save_confusion_matrix, save_history_plot, save_prediction_overlay
from .utils import json_dump, now_stamp, resolve_device, set_seed


CENTER_METHOD = "berrymtl_centerdet"
CENTER_DISPLAY = "BerryMTL-CenterDet (ours)"


class BerryMTLCenterDetNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        decoder_channels: int = 128,
        roi_channels: int = 192,
        roi_size: int = 7,
        detection_classes: int | None = None,
        decoupled_decoder: bool = False,
        dense_count_residual: bool = False,
        task_aligned_detection: bool = False,
        highres_detection: bool = False,
        roi_global_context: bool = False,
    ):
        super().__init__()
        import timm

        self.num_classes = int(num_classes)
        self.detection_classes = int(detection_classes if detection_classes is not None else num_classes - 1)
        self.roi_size = int(roi_size)
        self.decoupled_decoder = bool(decoupled_decoder)
        self.dense_count_residual = bool(dense_count_residual)
        self.task_aligned_detection = bool(task_aligned_detection)
        self.highres_detection = bool(highres_detection)
        self.roi_global_context = bool(roi_global_context)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels])
        decoder_in = decoder_channels * len(channels)
        self.shared = nn.Sequential(
            nn.Conv2d(decoder_in, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.08),
        )
        if self.decoupled_decoder:
            self.det_shared = nn.Sequential(
                nn.Conv2d(decoder_in, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
                nn.Dropout2d(0.08),
            )
            self.dense_shared = nn.Sequential(
                nn.Conv2d(decoder_in, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
                nn.Dropout2d(0.08),
            )
            self.dense_count_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(decoder_channels),
                nn.Linear(decoder_channels, 192),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(192, 1),
            )
            if self.dense_count_residual:
                nn.init.zeros_(self.dense_count_head[-1].weight)
                nn.init.zeros_(self.dense_count_head[-1].bias)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )
        self.roi_feature = nn.Sequential(
            nn.Conv2d(decoder_channels, roi_channels, kernel_size=1),
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
        self.roi_quality_head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(96, 1),
        )
        self.center_stem = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        if self.highres_detection:
            self.highres_det_proj = nn.Sequential(
                nn.Conv2d(channels[0], decoder_channels, kernel_size=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            )
            self.highres_det_fuse = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            )
        if self.task_aligned_detection:
            self.det_cls_stem = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            )
            self.det_reg_stem = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            )
            self.det_quality = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
                nn.Conv2d(decoder_channels, 1, kernel_size=1),
            )
        self.det_heatmap = nn.Conv2d(decoder_channels, self.detection_classes, kernel_size=1)
        self.det_size = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        self.det_offset = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        if self.roi_global_context:
            self.roi_context = nn.Sequential(
                nn.Conv2d(channels[-1], roi_channels, kernel_size=1),
                nn.BatchNorm2d(roi_channels),
                nn.GELU(),
            )
        self.density_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
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

    def classify_rois(
        self,
        roi_feature: torch.Tensor,
        seg_logits: torch.Tensor,
        boxes: list[torch.Tensor],
        image_size: int,
        return_features: bool = False,
        return_quality: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        rois = self._rois_from_boxes(boxes, roi_feature.device)
        if rois.numel() == 0:
            logits = torch.zeros((0, self.num_classes - 1), dtype=roi_feature.dtype, device=roi_feature.device)
            features = torch.zeros((0, 192), dtype=roi_feature.dtype, device=roi_feature.device)
            quality = torch.zeros((0,), dtype=roi_feature.dtype, device=roi_feature.device)
            if return_features and return_quality:
                return logits, features, quality
            if return_features:
                return logits, features
            if return_quality:
                return logits, quality
            return logits
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
        roi_tokens = self.class_head[:-1](pooled * (0.5 + attn))
        logits = self.class_head[-1](roi_tokens)
        quality = self.roi_quality_head(roi_tokens).squeeze(1)
        if return_features and return_quality:
            return logits, roi_tokens, quality
        if return_features:
            return logits, roi_tokens
        if return_quality:
            return logits, quality
        return logits

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
        decoder_input = torch.cat(projected, dim=1)
        if self.decoupled_decoder:
            dense_feature = self.dense_shared(decoder_input)
            det_feature = self.det_shared(decoder_input)
            count_pred = self.dense_count_head(dense_feature)
            if self.dense_count_residual:
                count_pred = self.count_head(features[-1]) + count_pred
        else:
            dense_feature = self.shared(decoder_input)
            det_feature = dense_feature
            count_pred = self.count_head(features[-1])
        seg_logits = self.seg_head(dense_feature)
        seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)
        density_logits = self.density_head(dense_feature)
        if self.highres_detection:
            highres_feature = self.highres_det_proj(features[0])
            det_feature = self.highres_det_fuse(torch.cat([det_feature, highres_feature], dim=1))
        center_feature = self.center_stem(det_feature)
        if self.task_aligned_detection:
            det_cls_feature = self.det_cls_stem(det_feature)
            det_reg_feature = self.det_reg_stem(det_feature)
            det_quality = self.det_quality(torch.cat([det_cls_feature, det_reg_feature], dim=1))
        else:
            det_cls_feature = center_feature
            det_reg_feature = center_feature
            det_quality = None
        roi_feature = self.roi_feature(det_feature)
        if self.roi_global_context:
            context = F.interpolate(self.roi_context(features[-1]), size=roi_feature.shape[-2:], mode="bilinear", align_corners=False)
            roi_feature = roi_feature + context
        output = {
            "seg": seg_logits,
            "count": count_pred,
            "density": density_logits,
            "det_heatmap": self.det_heatmap(det_cls_feature),
            "det_size": self.det_size(det_reg_feature),
            "det_offset": self.det_offset(det_reg_feature),
            "roi_feature": roi_feature,
        }
        if det_quality is not None:
            output["det_quality"] = det_quality
        if boxes is not None:
            cls_logits, cls_features, roi_quality = self.classify_rois(
                roi_feature,
                seg_logits,
                boxes,
                image_size=int(input_size[-1]),
                return_features=True,
                return_quality=True,
            )
            output["cls"] = cls_logits
            output["cls_features"] = cls_features
            output["roi_quality"] = roi_quality
        return output


def _draw_gaussian(target: torch.Tensor, center_x: float, center_y: float, radius: int) -> None:
    height, width = target.shape
    x_int = int(np.clip(np.floor(center_x), 0, width - 1))
    y_int = int(np.clip(np.floor(center_y), 0, height - 1))
    radius = max(1, int(radius))
    x1 = max(0, x_int - radius)
    x2 = min(width, x_int + radius + 1)
    y1 = max(0, y_int - radius)
    y2 = min(height, y_int + radius + 1)
    yy, xx = torch.meshgrid(
        torch.arange(y1, y2, device=target.device, dtype=torch.float32),
        torch.arange(x1, x2, device=target.device, dtype=torch.float32),
        indexing="ij",
    )
    sigma = max(float(radius) / 3.0, 1.0)
    patch = torch.exp(-((xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2) / (2.0 * sigma * sigma))
    target[y1:y2, x1:x2] = torch.maximum(target[y1:y2, x1:x2], patch.to(dtype=target.dtype))
    target[y_int, x_int] = 1.0


def _add_density(target: torch.Tensor, center_x: float, center_y: float, radius: int) -> None:
    height, width = target.shape
    x_int = int(np.clip(np.floor(center_x), 0, width - 1))
    y_int = int(np.clip(np.floor(center_y), 0, height - 1))
    radius = max(1, int(radius))
    x1 = max(0, x_int - radius)
    x2 = min(width, x_int + radius + 1)
    y1 = max(0, y_int - radius)
    y2 = min(height, y_int + radius + 1)
    yy, xx = torch.meshgrid(
        torch.arange(y1, y2, device=target.device, dtype=torch.float32),
        torch.arange(x1, x2, device=target.device, dtype=torch.float32),
        indexing="ij",
    )
    sigma = max(float(radius) / 2.0, 1.0)
    patch = torch.exp(-((xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2) / (2.0 * sigma * sigma))
    patch = patch / torch.clamp(patch.sum(), min=1e-6)
    target[y1:y2, x1:x2] += patch.to(dtype=target.dtype)


def _center_targets(
    boxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    image_size: int,
    heatmap_shape: tuple[int, int],
    num_classes: int,
    detection_classes: int,
    class_agnostic: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(boxes)
    heat_h, heat_w = heatmap_shape
    heatmap = torch.zeros((batch_size, detection_classes, heat_h, heat_w), dtype=torch.float32, device=device)
    size_target = torch.zeros((batch_size, 2, heat_h, heat_w), dtype=torch.float32, device=device)
    offset_target = torch.zeros((batch_size, 2, heat_h, heat_w), dtype=torch.float32, device=device)
    reg_mask = torch.zeros((batch_size, 1, heat_h, heat_w), dtype=torch.float32, device=device)
    density = torch.zeros((batch_size, 1, heat_h, heat_w), dtype=torch.float32, device=device)
    stride_x = float(image_size) / float(heat_w)
    stride_y = float(image_size) / float(heat_h)

    for batch_idx, (batch_boxes, batch_labels) in enumerate(zip(boxes, labels)):
        if batch_boxes.numel() == 0:
            continue
        batch_boxes = batch_boxes.to(device=device, dtype=torch.float32)
        batch_labels = batch_labels.to(device=device, dtype=torch.long)
        widths = torch.clamp(batch_boxes[:, 2] - batch_boxes[:, 0], min=1.0)
        heights = torch.clamp(batch_boxes[:, 3] - batch_boxes[:, 1], min=1.0)
        centers_x = (batch_boxes[:, 0] + batch_boxes[:, 2]) * 0.5 / stride_x
        centers_y = (batch_boxes[:, 1] + batch_boxes[:, 3]) * 0.5 / stride_y
        for box_idx in range(batch_boxes.shape[0]):
            cls = 0 if class_agnostic else int(batch_labels[box_idx].item())
            if cls < 0 or cls >= detection_classes:
                continue
            center_x = float(centers_x[box_idx].clamp(0, heat_w - 1).item())
            center_y = float(centers_y[box_idx].clamp(0, heat_h - 1).item())
            x_int = int(np.clip(np.floor(center_x), 0, heat_w - 1))
            y_int = int(np.clip(np.floor(center_y), 0, heat_h - 1))
            width = float(widths[box_idx].item())
            height = float(heights[box_idx].item())
            radius = int(max(1.0, min(width / stride_x, height / stride_y) * 0.35))
            _draw_gaussian(heatmap[batch_idx, cls], center_x, center_y, radius)
            _add_density(density[batch_idx, 0], center_x, center_y, radius)
            size_target[batch_idx, 0, y_int, x_int] = width
            size_target[batch_idx, 1, y_int, x_int] = height
            offset_target[batch_idx, 0, y_int, x_int] = center_x - float(x_int)
            offset_target[batch_idx, 1, y_int, x_int] = center_y - float(y_int)
            reg_mask[batch_idx, 0, y_int, x_int] = 1.0
    return {
        "heatmap": heatmap,
        "size": size_target,
        "offset": offset_target,
        "reg_mask": reg_mask,
        "density": density,
    }


def _focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.eq(1.0).float()
    neg = target.lt(1.0).float()
    neg_weights = torch.pow(1.0 - target, 4.0)
    pos_loss = torch.log(prob) * torch.pow(1.0 - prob, 2.0) * pos
    neg_loss = torch.log(1.0 - prob) * torch.pow(prob, 2.0) * neg_weights * neg
    num_pos = torch.clamp(pos.sum(), min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def _masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    denom = torch.clamp(expanded.sum(), min=1.0)
    return F.l1_loss(prediction * expanded, target * expanded, reduction="sum") / denom


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.numel() == 0 or labels.numel() < 2:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    features = F.normalize(features.float(), dim=1)
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(features.device)
    self_mask = torch.eye(labels.shape[0], dtype=torch.float32, device=features.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    valid = positive_mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return torch.zeros((), dtype=features.dtype, device=features.device)
    logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1.0 - self_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-9))
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, gamma: float, label_smoothing: float):
        super().__init__()
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.detach().clone().float())
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probs = torch.softmax(logits.float(), dim=1)
        pt = probs.gather(1, target.view(-1, 1)).squeeze(1).clamp(min=1e-4, max=1.0 - 1e-4)
        return (torch.pow(1.0 - pt, self.gamma) * ce).mean()


def _center_detection_loss(output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    heat_loss = _focal_heatmap_loss(output["det_heatmap"].float(), targets["heatmap"])
    size_pred = F.softplus(output["det_size"].float())
    offset_pred = torch.sigmoid(output["det_offset"].float())
    size_loss = _masked_l1_loss(size_pred, targets["size"], targets["reg_mask"])
    offset_loss = _masked_l1_loss(offset_pred, targets["offset"], targets["reg_mask"])
    quality_loss = torch.zeros((), dtype=heat_loss.dtype, device=heat_loss.device)
    if "det_quality" in output:
        quality_target = targets["heatmap"].amax(dim=1, keepdim=True)
        quality_loss = _focal_heatmap_loss(output["det_quality"].float(), quality_target)
    total = heat_loss + 0.5 * quality_loss + 0.04 * size_loss + offset_loss
    parts = {
        "det_heatmap_loss": float(heat_loss.detach().cpu().item()),
        "det_quality_loss": float(quality_loss.detach().cpu().item()),
        "det_size_loss": float(size_loss.detach().cpu().item()),
        "det_offset_loss": float(offset_loss.detach().cpu().item()),
    }
    return total, parts


def _aligned_box_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0],), dtype=boxes1.dtype, device=boxes1.device)
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = (area1 + area2 - inter).clamp_min(1e-6)
    return inter / union


def _max_box_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0:
        return torch.zeros((0,), dtype=boxes1.dtype, device=boxes1.device)
    if boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0],), dtype=boxes1.dtype, device=boxes1.device)
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = (area1[:, None] + area2[None, :] - inter).clamp_min(1e-6)
    return (inter / union).amax(dim=1)


def _roi_quality_targets(
    boxes: list[torch.Tensor],
    image_size: int,
    jitter_count: int,
    background_count: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    quality_boxes: list[torch.Tensor] = []
    quality_targets: list[torch.Tensor] = []
    for batch_boxes in boxes:
        device = batch_boxes.device
        dtype = batch_boxes.dtype
        batch_boxes = batch_boxes.to(dtype=torch.float32)
        per_image_boxes: list[torch.Tensor] = []
        per_image_targets: list[torch.Tensor] = []
        if batch_boxes.numel() > 0:
            per_image_boxes.append(batch_boxes)
            per_image_targets.append(torch.ones((batch_boxes.shape[0],), dtype=torch.float32, device=device))
            widths = (batch_boxes[:, 2] - batch_boxes[:, 0]).clamp_min(2.0)
            heights = (batch_boxes[:, 3] - batch_boxes[:, 1]).clamp_min(2.0)
            centers_x = (batch_boxes[:, 0] + batch_boxes[:, 2]) * 0.5
            centers_y = (batch_boxes[:, 1] + batch_boxes[:, 3]) * 0.5
            for _ in range(max(0, int(jitter_count))):
                shift = (torch.rand((batch_boxes.shape[0], 2), device=device) - 0.5) * 0.70
                scale = 0.65 + torch.rand((batch_boxes.shape[0], 2), device=device) * 0.85
                new_w = (widths * scale[:, 0]).clamp_min(2.0)
                new_h = (heights * scale[:, 1]).clamp_min(2.0)
                new_cx = centers_x + shift[:, 0] * widths
                new_cy = centers_y + shift[:, 1] * heights
                jittered = torch.stack(
                    [
                        (new_cx - new_w * 0.5).clamp(0, image_size),
                        (new_cy - new_h * 0.5).clamp(0, image_size),
                        (new_cx + new_w * 0.5).clamp(0, image_size),
                        (new_cy + new_h * 0.5).clamp(0, image_size),
                    ],
                    dim=1,
                )
                valid = (jittered[:, 2] - jittered[:, 0] >= 2.0) & (jittered[:, 3] - jittered[:, 1] >= 2.0)
                if bool(valid.any()):
                    jittered = jittered[valid]
                    source = batch_boxes[valid]
                    per_image_boxes.append(jittered)
                    per_image_targets.append(_aligned_box_iou_torch(jittered, source).detach())
        if background_count > 0:
            count = int(background_count)
            side_min = max(4.0, float(image_size) * 0.025)
            side_max = max(side_min + 1.0, float(image_size) * 0.16)
            wh = side_min + torch.rand((count, 2), device=device) * (side_max - side_min)
            xy1 = torch.rand((count, 2), device=device) * (float(image_size) - wh).clamp_min(1.0)
            random_boxes = torch.cat([xy1, xy1 + wh], dim=1)
            per_image_boxes.append(random_boxes)
            per_image_targets.append(_max_box_iou_torch(random_boxes, batch_boxes).detach())
        if per_image_boxes:
            quality_boxes.append(torch.cat(per_image_boxes, dim=0).to(dtype=dtype))
            quality_targets.append(torch.cat(per_image_targets, dim=0).clamp(0.0, 1.0))
        else:
            quality_boxes.append(torch.zeros((0, 4), dtype=dtype, device=device))
    if not quality_targets:
        device = boxes[0].device if boxes else torch.device("cpu")
        return quality_boxes, torch.zeros((0,), dtype=torch.float32, device=device)
    return quality_boxes, torch.cat(quality_targets, dim=0)


def _density_losses(
    output: dict[str, torch.Tensor],
    target_density: torch.Tensor,
    count_target: torch.Tensor,
    count_loss: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_density = F.softplus(output["density"].float())
    map_loss = F.mse_loss(pred_density, target_density)
    density_count = pred_density.sum(dim=(1, 2, 3), keepdim=False).view(-1, 1)
    count_from_density = count_loss(density_count, count_target.float())
    return map_loss, count_from_density, density_count


@torch.inference_mode()
def _decode_center_detections(
    output: dict[str, torch.Tensor],
    image_size: int,
    score_threshold: float,
    top_k: int,
    nms_iou: float,
    class_agnostic: bool = False,
) -> list[dict[str, np.ndarray]]:
    heatmap = torch.sigmoid(output["det_heatmap"].float())
    if "det_quality" in output:
        heatmap = heatmap * torch.sigmoid(output["det_quality"].float())
    size = F.softplus(output["det_size"].float())
    offset = torch.sigmoid(output["det_offset"].float())
    batch_size, channels, heat_h, heat_w = heatmap.shape
    stride_x = float(image_size) / float(heat_w)
    stride_y = float(image_size) / float(heat_h)
    keep = heatmap.eq(F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1))
    heatmap = heatmap * keep
    decoded: list[dict[str, np.ndarray]] = []
    for batch_idx in range(batch_size):
        scores, indices = torch.topk(heatmap[batch_idx].reshape(-1), k=min(top_k, channels * heat_h * heat_w))
        valid = scores >= score_threshold
        scores = scores[valid]
        indices = indices[valid]
        if indices.numel() == 0:
            decoded.append(
                {
                    "boxes": np.zeros((0, 4), dtype=float),
                    "labels": np.zeros((0,), dtype=int),
                    "scores": np.zeros((0,), dtype=float),
                }
            )
            continue
        cls = indices // (heat_h * heat_w)
        rem = indices % (heat_h * heat_w)
        ys = rem // heat_w
        xs = rem % heat_w
        local_size = size[batch_idx, :, ys, xs].transpose(0, 1)
        local_offset = offset[batch_idx, :, ys, xs].transpose(0, 1)
        center_x = (xs.float() + local_offset[:, 0]) * stride_x
        center_y = (ys.float() + local_offset[:, 1]) * stride_y
        widths = local_size[:, 0].clamp(min=2.0, max=float(image_size))
        heights = local_size[:, 1].clamp(min=2.0, max=float(image_size))
        boxes = torch.stack(
            [
                (center_x - widths * 0.5).clamp(0, image_size),
                (center_y - heights * 0.5).clamp(0, image_size),
                (center_x + widths * 0.5).clamp(0, image_size),
                (center_y + heights * 0.5).clamp(0, image_size),
            ],
            dim=1,
        )
        labels = torch.ones_like(cls, dtype=torch.long) if class_agnostic else cls.to(torch.long) + 1
        keep_indices = []
        for class_id in labels.unique():
            selected = torch.where(labels == class_id)[0]
            kept = nms(boxes[selected], scores[selected], nms_iou)
            keep_indices.append(selected[kept])
        keep_all = torch.cat(keep_indices, dim=0) if keep_indices else torch.zeros((0,), dtype=torch.long, device=boxes.device)
        keep_all = keep_all[torch.argsort(scores[keep_all], descending=True)]
        decoded.append(
            {
                "boxes": boxes[keep_all].detach().cpu().numpy().astype(float).reshape(-1, 4),
                "labels": labels[keep_all].detach().cpu().numpy().astype(int),
                "scores": scores[keep_all].detach().cpu().numpy().astype(float),
            }
        )
    return decoded


def _foreground_support_for_boxes(foreground: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=float)
    height, width = foreground.shape
    supports: list[float] = []
    for box in boxes:
        x1 = int(np.floor(np.clip(box[0], 0, width - 1)))
        y1 = int(np.floor(np.clip(box[1], 0, height - 1)))
        x2 = int(np.ceil(np.clip(box[2], x1 + 1, width)))
        y2 = int(np.ceil(np.clip(box[3], y1 + 1, height)))
        crop = foreground[y1:y2, x1:x2]
        if crop.size == 0:
            supports.append(0.0)
            continue
        center_x = int(np.clip(round((float(box[0]) + float(box[2])) * 0.5), 0, width - 1))
        center_y = int(np.clip(round((float(box[1]) + float(box[3])) * 0.5), 0, height - 1))
        supports.append(float(0.65 * crop.mean() + 0.35 * foreground[center_y, center_x]))
    return np.asarray(supports, dtype=float)


def _refine_boxes_with_foreground(
    foreground: np.ndarray,
    boxes: np.ndarray,
    threshold: float,
    expansion: float,
    blend: float,
    min_pixels: int,
) -> np.ndarray:
    if len(boxes) == 0 or blend <= 0:
        return boxes.astype(float, copy=False).reshape(-1, 4)
    height, width = foreground.shape
    refined = boxes.astype(float, copy=True)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * float(expansion)
        pad_y = bh * float(expansion)
        sx1 = int(np.floor(np.clip(x1 - pad_x, 0, width - 1)))
        sy1 = int(np.floor(np.clip(y1 - pad_y, 0, height - 1)))
        sx2 = int(np.ceil(np.clip(x2 + pad_x, sx1 + 1, width)))
        sy2 = int(np.ceil(np.clip(y2 + pad_y, sy1 + 1, height)))
        crop = foreground[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue
        mask = crop >= float(threshold)
        if int(mask.sum()) < int(min_pixels):
            continue
        ys, xs = np.where(mask)
        mask_box = np.asarray(
            [
                sx1 + float(xs.min()),
                sy1 + float(ys.min()),
                sx1 + float(xs.max() + 1),
                sy1 + float(ys.max() + 1),
            ],
            dtype=float,
        )
        mask_w = max(1.0, mask_box[2] - mask_box[0])
        mask_h = max(1.0, mask_box[3] - mask_box[1])
        # Avoid letting nearby berries in a dense cluster explode a box into a branch-sized region.
        if mask_w > 2.8 * bw or mask_h > 2.8 * bh:
            continue
        mixed = (1.0 - float(blend)) * np.asarray([x1, y1, x2, y2], dtype=float) + float(blend) * mask_box
        mixed[0::2] = np.clip(mixed[0::2], 0, width)
        mixed[1::2] = np.clip(mixed[1::2], 0, height)
        if mixed[2] - mixed[0] >= 2.0 and mixed[3] - mixed[1] >= 2.0:
            refined[idx] = mixed
    return refined.astype(float, copy=False).reshape(-1, 4)


def _apply_quality_detection_context(
    pred_det: dict[str, np.ndarray],
    foreground: np.ndarray,
    count_prediction: float,
    segmentation_support_power: float,
    segmentation_support_threshold: float,
    segmentation_box_refine: bool,
    segmentation_box_refine_threshold: float,
    segmentation_box_refine_expansion: float,
    segmentation_box_refine_blend: float,
    segmentation_box_refine_min_pixels: int,
    count_aware_topk: bool,
    count_aware_multiplier: float,
    count_aware_bias: float,
    count_aware_min: int,
    count_aware_max: int,
) -> dict[str, np.ndarray]:
    if len(pred_det["boxes"]) == 0:
        return pred_det
    boxes = pred_det["boxes"]
    labels = pred_det["labels"]
    scores = pred_det["scores"].astype(float, copy=True)

    support = _foreground_support_for_boxes(foreground, boxes)
    if segmentation_support_power > 0:
        scores *= np.clip(support, 1e-4, 1.0) ** float(segmentation_support_power)
    keep = np.ones((len(scores),), dtype=bool)
    if segmentation_support_threshold > 0:
        keep &= support >= float(segmentation_support_threshold)
    if keep.sum() != len(keep):
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        support = support[keep]
    if segmentation_box_refine and len(scores) > 0:
        boxes = _refine_boxes_with_foreground(
            foreground,
            boxes,
            threshold=segmentation_box_refine_threshold,
            expansion=segmentation_box_refine_expansion,
            blend=segmentation_box_refine_blend,
            min_pixels=segmentation_box_refine_min_pixels,
        )

    if count_aware_topk and len(scores) > 0:
        cap = int(np.ceil(max(float(count_aware_min), float(count_prediction) * float(count_aware_multiplier) + float(count_aware_bias))))
        if count_aware_max > 0:
            cap = min(cap, int(count_aware_max))
        cap = max(1, min(cap, len(scores)))
        order = np.argsort(-scores)[:cap]
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
        support = support[order]

    return {
        "boxes": boxes.astype(float, copy=False).reshape(-1, 4),
        "labels": labels.astype(int, copy=False),
        "scores": scores.astype(float, copy=False),
        "foreground_support": support.astype(float, copy=False),
    }


def _center_epoch(
    model: BerryMTLCenterDetNet,
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
    image_size: int,
    count_weight: float,
    dice_weight: float,
    cls_weight: float,
    det_weight: float,
    density_weight: float,
    density_count_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    roi_quality_weight: float,
    roi_quality_jitter_count: int,
    roi_quality_background_count: int,
    cls_score_weight: float,
    count_score_weight: float,
    det_score_weight: float,
    detection_classes: int,
    class_agnostic_detection: bool,
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
    det_loss_total = 0.0
    density_loss_total = 0.0
    contrastive_loss_total = 0.0
    roi_quality_loss_total = 0.0
    image_count = 0
    roi_count = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")

    for images, masks, targets, _, boxes, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        boxes = [value.to(device, non_blocking=True) for value in boxes]
        labels = [value.to(device, non_blocking=True) for value in labels]
        cls_labels = _flatten_labels(labels, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images, boxes=boxes)
            logits = output["seg"]
            counts = output["count"].float()
            center_targets = _center_targets(
                boxes,
                labels,
                image_size=image_size,
                heatmap_shape=output["det_heatmap"].shape[-2:],
                num_classes=num_classes,
                detection_classes=detection_classes,
                class_agnostic=class_agnostic_detection,
                device=device,
            )
            seg_loss = seg_criterion(logits, masks) + dice_weight * _dice_loss(logits, masks, num_classes)
            c_loss = count_loss(counts, targets.float())
            d_loss, det_parts = _center_detection_loss(output, center_targets)
            density_map_loss, density_count_loss, _ = _density_losses(output, center_targets["density"], targets.float(), count_loss)
            density_loss = density_map_loss + density_count_weight * density_count_loss
            if cls_labels.numel() > 0:
                cls_logits = output["cls"]
                r_loss = cls_criterion(cls_logits, cls_labels)
                contrastive_loss = _supervised_contrastive_loss(output["cls_features"], cls_labels, contrastive_temperature)
            else:
                cls_logits = torch.zeros((0, num_classes - 1), device=device)
                r_loss = torch.zeros((), device=device)
                contrastive_loss = torch.zeros((), device=device)
            roi_quality_loss = torch.zeros((), device=device)
            if roi_quality_weight > 0:
                quality_boxes, quality_targets = _roi_quality_targets(
                    boxes,
                    image_size=image_size,
                    jitter_count=roi_quality_jitter_count,
                    background_count=roi_quality_background_count,
                )
                if quality_targets.numel() > 0:
                    _, _, quality_logits = model.classify_rois(
                        output["roi_feature"],
                        logits,
                        quality_boxes,
                        image_size=image_size,
                        return_features=True,
                        return_quality=True,
                    )
                    roi_quality_loss = F.binary_cross_entropy_with_logits(
                        quality_logits.float().view(-1),
                        quality_targets.to(device=quality_logits.device, dtype=torch.float32).view(-1),
                    )
            loss = (
                seg_loss
                + count_weight * c_loss
                + cls_weight * r_loss
                + det_weight * d_loss
                + density_weight * density_loss
                + contrastive_weight * contrastive_loss
                + roi_quality_weight * roi_quality_loss
            )
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
        det_loss_total += float(d_loss.item()) * batch
        density_loss_total += float(density_loss.item()) * batch
        contrastive_loss_total += float(contrastive_loss.item()) * max(1, int(cls_labels.numel()))
        roi_quality_loss_total += float(roi_quality_loss.item()) * batch
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
    det_loss_avg = det_loss_total / max(1, image_count)
    output_metrics = {
        "loss": total_loss / max(1, image_count),
        "seg_loss": seg_loss_total / max(1, image_count),
        "count_loss": count_loss_total / max(1, image_count),
        "roi_cls_loss": cls_loss_total / max(1, roi_count),
        "center_det_loss": det_loss_avg,
        "density_loss": density_loss_total / max(1, image_count),
        "contrastive_loss": contrastive_loss_total / max(1, roi_count),
        "roi_quality_loss": roi_quality_loss_total / max(1, image_count),
        **{f"seg_{key}": value for key, value in seg_metrics.items()},
        **{f"count_{key}": value for key, value in c_metrics.items()},
        **{f"cls_{key}": value for key, value in cls_metrics.items()},
    }
    output_metrics["joint_score"] = float(
        seg_metrics["miou_foreground"]
        + cls_score_weight * cls_metrics["macro_f1"]
        - count_score_weight * c_metrics["mae"]
        - det_score_weight * det_loss_avg
    )
    return output_metrics


@torch.inference_mode()
def _evaluate_center(
    model: BerryMTLCenterDetNet,
    loader: DataLoader,
    image_frame: pd.DataFrame,
    instances: pd.DataFrame,
    device: torch.device,
    amp: bool,
    image_size: int,
    class_names: list[str],
    score_threshold: float,
    top_k: int,
    nms_iou: float,
    classify_detection_boxes: bool,
    class_agnostic_detection: bool,
    segmentation_support_power: float = 0.0,
    segmentation_support_threshold: float = 0.0,
    segmentation_box_refine: bool = False,
    segmentation_box_refine_threshold: float = 0.30,
    segmentation_box_refine_expansion: float = 0.10,
    segmentation_box_refine_blend: float = 0.30,
    segmentation_box_refine_min_pixels: int = 6,
    count_aware_topk: bool = False,
    count_aware_multiplier: float = 1.25,
    count_aware_bias: float = 12.0,
    count_aware_min: int = 20,
    count_aware_max: int = 0,
    roi_quality_inference_power: float = 0.0,
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
        labels = [value.to(device, non_blocking=True) for value in labels]
        cls_labels = _flatten_labels(labels, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images, boxes=boxes)
            logits = output["seg"]
            counts = output["count"].float()
            cls_logits = output["cls"]
        pred_batch = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        foreground_batch = torch.softmax(logits.detach().float(), dim=1)[:, 1:].sum(dim=1).cpu().numpy().astype(float)
        true_batch = masks.detach().cpu().numpy().astype(np.uint8)
        seg_meter.update(logits, masks)
        count_true.extend(targets.numpy().reshape(-1).tolist())
        count_pred.extend(counts.detach().cpu().numpy().reshape(-1).tolist())
        pred_detections = _decode_center_detections(output, image_size, score_threshold, top_k, nms_iou, class_agnostic=class_agnostic_detection)

        if classify_detection_boxes and any(len(value["boxes"]) > 0 for value in pred_detections):
            det_boxes_for_batch = [torch.as_tensor(value["boxes"], dtype=torch.float32, device=device) for value in pred_detections]
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                det_logits, det_quality_logits = model.classify_rois(
                    output["roi_feature"],
                    logits,
                    det_boxes_for_batch,
                    image_size=image_size,
                    return_quality=True,
                )
            det_probs = torch.softmax(det_logits.detach().float(), dim=1).cpu().numpy() if det_logits.numel() else np.zeros((0, len(class_names)))
            det_quality_scores = (
                torch.sigmoid(det_quality_logits.detach().float()).cpu().numpy()
                if det_quality_logits.numel()
                else np.zeros((0,), dtype=float)
            )
        else:
            det_probs = np.zeros((0, len(class_names)))
            det_quality_scores = np.zeros((0,), dtype=float)

        cls_offset = 0
        det_offset = 0
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

            pred_det = pred_detections[local_idx]
            if classify_detection_boxes and len(pred_det["boxes"]) > 0 and det_probs.size:
                local_probs = det_probs[det_offset : det_offset + len(pred_det["boxes"])]
                local_quality = det_quality_scores[det_offset : det_offset + len(pred_det["boxes"])]
                det_offset += len(pred_det["boxes"])
                class_pred = local_probs.argmax(axis=1) + 1
                class_conf = local_probs.max(axis=1)
                scores = pred_det["scores"] * class_conf
                if roi_quality_inference_power > 0:
                    scores = scores * np.clip(local_quality, 1e-4, 1.0) ** float(roi_quality_inference_power)
                pred_det = {
                    "boxes": pred_det["boxes"],
                    "labels": class_pred.astype(int),
                    "scores": scores.astype(float),
                }
            pred_det = _apply_quality_detection_context(
                pred_det,
                foreground=foreground_batch[local_idx],
                count_prediction=float(counts[local_idx].detach().cpu().item()),
                segmentation_support_power=segmentation_support_power,
                segmentation_support_threshold=segmentation_support_threshold,
                segmentation_box_refine=segmentation_box_refine,
                segmentation_box_refine_threshold=segmentation_box_refine_threshold,
                segmentation_box_refine_expansion=segmentation_box_refine_expansion,
                segmentation_box_refine_blend=segmentation_box_refine_blend,
                segmentation_box_refine_min_pixels=segmentation_box_refine_min_pixels,
                count_aware_topk=count_aware_topk,
                count_aware_multiplier=count_aware_multiplier,
                count_aware_bias=count_aware_bias,
                count_aware_min=count_aware_min,
                count_aware_max=count_aware_max,
            )
            target_det = instance_targets[stem]
            detection_predictions.append(pred_det)
            detection_targets.append(target_det)
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
            if sample_payload is None:
                sample_payload = (row["image_path"], true_mask, pred_mask)

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
        "family": "Unified multitask anchor-free detection",
        "resolved_model_name": f"timm_fpn_centernet:{model_name}",
        "seed": seed,
        "device": str(device),
        "parameters": int(parameters),
        "trainable_parameters": int(parameters),
        "image_size": int(image_size),
        "shared_backbone": True,
        "center_detection_head": True,
        "density_count_head": True,
        "roi_attention": True,
        "classification_head": "mask-attended ROI head",
    }


def _image_sampling_weights(image_frame: pd.DataFrame, instances: pd.DataFrame, power: float) -> torch.Tensor:
    if power <= 0 or instances.empty:
        return torch.ones((len(image_frame),), dtype=torch.float32)
    counts = instances["class_index"].value_counts().to_dict()
    max_count = max([float(value) for value in counts.values()] or [1.0])
    class_weights = {
        int(label): float(np.clip((max_count / max(float(count), 1.0)) ** float(power), 1.0, 16.0))
        for label, count in counts.items()
    }
    by_stem = {
        str(stem): group["class_index"].to_numpy(dtype=np.int64, copy=True)
        for stem, group in instances.groupby("stem", sort=False)
    }
    weights = []
    for stem in image_frame["stem"].astype(str):
        labels = by_stem.get(stem)
        if labels is None or len(labels) == 0:
            weights.append(1.0)
            continue
        local = np.asarray([class_weights.get(int(label), 1.0) for label in labels], dtype=np.float64)
        weights.append(float(0.5 * local.max() + 0.5 * local.mean()))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.mean()), 1e-9)
    return torch.as_tensor(weights, dtype=torch.float32)


def run_berrymtl_centerdet(
    config: dict[str, Any],
    seed: int | None = None,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
    profile_key: str = "ours_centerdet",
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    seed = int(seed if seed is not None else config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(device_name)
    classes = list(config["classes"])
    num_classes = len(classes) + 1
    cfg = {**config.get("ours", {}), **config.get(profile_key, {})}
    method = str(cfg.get("method", CENTER_METHOD))
    display_name = str(cfg.get("display_name", CENTER_DISPLAY))
    model_name = str(cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))
    image_size = int(cfg.get("image_size", 512))
    batch_size = int(batch_size if batch_size is not None else cfg.get("batch_size", 3))
    epochs = int(epochs if epochs is not None else cfg.get("epochs", 60))
    count_weight = float(cfg.get("count_loss_weight", 0.035))
    dice_weight = float(cfg.get("dice_loss_weight", 0.6))
    cls_weight = float(cfg.get("classification_loss_weight", 0.55))
    det_weight = float(cfg.get("detection_loss_weight", 0.45))
    density_weight = float(cfg.get("density_loss_weight", 0.02))
    density_count_weight = float(cfg.get("density_count_loss_weight", 0.15))
    contrastive_weight = float(cfg.get("contrastive_loss_weight", 0.0))
    contrastive_temperature = float(cfg.get("contrastive_temperature", 0.12))
    cls_score_weight = float(cfg.get("classification_score_weight", 0.25))
    count_score_weight = float(cfg.get("count_score_weight", 0.002))
    det_score_weight = float(cfg.get("detection_score_weight", 0.01))
    score_threshold = float(cfg.get("score_threshold", 0.05))
    top_k = int(cfg.get("top_k", 300))
    nms_iou = float(cfg.get("nms_iou", 0.45))
    classify_detection_boxes = bool(cfg.get("classify_detection_boxes", True))
    class_agnostic_detection = bool(cfg.get("class_agnostic_detection", False))
    detection_classes = 1 if class_agnostic_detection else len(classes)
    decoupled_decoder = bool(cfg.get("decoupled_decoder", False))
    dense_count_residual = bool(cfg.get("dense_count_residual", False))
    task_aligned_detection = bool(cfg.get("task_aligned_detection", False))
    highres_detection = bool(cfg.get("highres_detection", False))
    roi_global_context = bool(cfg.get("roi_global_context", False))
    train_tile_prob = float(cfg.get("train_tile_prob", 0.0))
    train_tile_min_scale = float(cfg.get("train_tile_min_scale", 0.42))
    train_tile_max_scale = float(cfg.get("train_tile_max_scale", 0.72))
    train_tile_min_visibility = float(cfg.get("train_tile_min_visibility", 0.25))
    train_tile_anchor_class_power = float(cfg.get("train_tile_anchor_class_power", 0.0))
    train_image_sampling_power = float(cfg.get("train_image_sampling_power", 0.0))
    roi_focal_gamma = float(cfg.get("roi_focal_gamma", 0.0))
    roi_class_weight_power = float(cfg.get("roi_class_weight_power", 1.0))
    roi_quality_weight = float(cfg.get("roi_quality_loss_weight", 0.0))
    roi_quality_jitter_count = int(cfg.get("roi_quality_jitter_count", 0))
    roi_quality_background_count = int(cfg.get("roi_quality_background_count", 0))
    roi_quality_inference_power = float(cfg.get("roi_quality_inference_power", 0.0))
    segmentation_support_power = float(cfg.get("segmentation_support_power", 0.0))
    segmentation_support_threshold = float(cfg.get("segmentation_support_threshold", 0.0))
    segmentation_box_refine = bool(cfg.get("segmentation_box_refine", False))
    segmentation_box_refine_threshold = float(cfg.get("segmentation_box_refine_threshold", 0.30))
    segmentation_box_refine_expansion = float(cfg.get("segmentation_box_refine_expansion", 0.10))
    segmentation_box_refine_blend = float(cfg.get("segmentation_box_refine_blend", 0.30))
    segmentation_box_refine_min_pixels = int(cfg.get("segmentation_box_refine_min_pixels", 6))
    count_aware_topk = bool(cfg.get("count_aware_topk", False))
    count_aware_multiplier = float(cfg.get("count_aware_multiplier", 1.25))
    count_aware_bias = float(cfg.get("count_aware_bias", 12.0))
    count_aware_min = int(cfg.get("count_aware_min", 20))
    count_aware_max = int(cfg.get("count_aware_max", top_k))
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
    train_sampler = None
    shuffle_train = True
    if train_image_sampling_power > 0:
        sample_weights = _image_sampling_weights(train_images, train_instances, train_image_sampling_power)
        train_sampler = WeightedRandomSampler(
            sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
    train_loader = DataLoader(
        BerryMTLInstanceDataset(
            train_images,
            train_instances,
            image_size,
            True,
            tile_prob=train_tile_prob,
            tile_min_scale=train_tile_min_scale,
            tile_max_scale=train_tile_max_scale,
            tile_min_visibility=train_tile_min_visibility,
            tile_anchor_class_power=train_tile_anchor_class_power,
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
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

    model = BerryMTLCenterDetNet(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        decoder_channels=int(cfg.get("decoder_channels", 128)),
        roi_channels=int(cfg.get("roi_channels", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
        detection_classes=detection_classes,
        decoupled_decoder=decoupled_decoder,
        dense_count_residual=dense_count_residual,
        task_aligned_detection=task_aligned_detection,
        highres_detection=highres_detection,
        roi_global_context=roi_global_context,
    ).to(device)
    init_source = None
    if bool(cfg.get("init_from_centerdet", False)):
        source_root = dirs["analysis"] / "ours" / "runs"
        patterns = cfg.get("init_source_patterns")
        if not patterns:
            patterns = ["*_berrymtl_centerdet_calibrated_seed*", "*_berrymtl_centerdet_seed*"]
        candidates = []
        for pattern in patterns:
            candidates.extend(source_root.glob(str(pattern)))
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        candidates = [path for path in candidates if (path / "best.pt").exists()]
        if not candidates:
            raise FileNotFoundError(f"No BerryMTL CenterDet checkpoint found under {source_root}")
        init_source = candidates[0] / "best.pt"
        checkpoint = torch.load(init_source, map_location=device, weights_only=False)
        source_state = checkpoint["model"]
        current_state = model.state_dict()
        if decoupled_decoder:
            expanded_state = dict(source_state)
            for branch_name in ["det_shared", "dense_shared"]:
                for key, value in source_state.items():
                    if key.startswith("shared."):
                        expanded_state[f"{branch_name}.{key[len('shared.'):]}"] = value
            source_state = expanded_state
        if task_aligned_detection:
            expanded_state = dict(source_state)
            for branch_name in ["det_cls_stem", "det_reg_stem"]:
                for key, value in source_state.items():
                    if key.startswith("center_stem."):
                        expanded_state[f"{branch_name}.{key[len('center_stem.'):]}"] = value
            source_state = expanded_state
        compatible = {key: value for key, value in source_state.items() if key in current_state and current_state[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
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
    if abs(roi_class_weight_power - 1.0) > 1e-6:
        cls_weights = torch.pow(cls_weights, roi_class_weight_power)
        cls_weights = cls_weights / torch.clamp(cls_weights.mean(), min=1e-9)
        cls_weights = torch.clamp(cls_weights, min=0.20, max=12.0)
    seg_loss = nn.CrossEntropyLoss(weight=seg_weights)
    label_smoothing = float(cfg.get("label_smoothing", 0.03))
    if roi_focal_gamma > 0:
        cls_loss = FocalCrossEntropyLoss(cls_weights, gamma=roi_focal_gamma, label_smoothing=label_smoothing)
    else:
        cls_loss = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
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
        train_metrics = _center_epoch(
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
            image_size,
            count_weight,
            dice_weight,
            cls_weight,
            det_weight,
            density_weight,
            density_count_weight,
            contrastive_weight,
            contrastive_temperature,
            roi_quality_weight,
            roi_quality_jitter_count,
            roi_quality_background_count,
            cls_score_weight,
            count_score_weight,
            det_score_weight,
            detection_classes,
            class_agnostic_detection,
        )
        val_metrics = _center_epoch(
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
            image_size,
            count_weight,
            dice_weight,
            cls_weight,
            det_weight,
            density_weight,
            density_count_weight,
            contrastive_weight,
            contrastive_temperature,
            roi_quality_weight,
            roi_quality_jitter_count,
            roi_quality_background_count,
            cls_score_weight,
            count_score_weight,
            det_score_weight,
            detection_classes,
            class_agnostic_detection,
        )
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}}
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
    test = _evaluate_center(
        model,
        test_loader,
        test_images,
        test_instances,
        device,
        amp,
        image_size,
        classes,
        score_threshold=score_threshold,
        top_k=top_k,
        nms_iou=nms_iou,
        classify_detection_boxes=classify_detection_boxes,
        class_agnostic_detection=class_agnostic_detection,
        segmentation_support_power=segmentation_support_power,
        segmentation_support_threshold=segmentation_support_threshold,
        segmentation_box_refine=segmentation_box_refine,
        segmentation_box_refine_threshold=segmentation_box_refine_threshold,
        segmentation_box_refine_expansion=segmentation_box_refine_expansion,
        segmentation_box_refine_blend=segmentation_box_refine_blend,
        segmentation_box_refine_min_pixels=segmentation_box_refine_min_pixels,
        count_aware_topk=count_aware_topk,
        count_aware_multiplier=count_aware_multiplier,
        count_aware_bias=count_aware_bias,
        count_aware_min=count_aware_min,
        count_aware_max=count_aware_max,
        roi_quality_inference_power=roi_quality_inference_power,
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
            "detection_loss_weight": det_weight,
            "density_loss_weight": density_weight,
            "contrastive_loss_weight": contrastive_weight,
            "contrastive_temperature": contrastive_temperature,
            "score_threshold": score_threshold,
            "top_k": top_k,
            "nms_iou": nms_iou,
            "classify_detection_boxes": classify_detection_boxes,
            "class_agnostic_detection": class_agnostic_detection,
            "detection_classes": detection_classes,
            "decoupled_decoder": decoupled_decoder,
            "decoder_branching": "det_cls_vs_seg_count" if decoupled_decoder else "shared",
            "dense_count_residual": dense_count_residual,
            "task_aligned_detection": task_aligned_detection,
            "highres_detection": highres_detection,
            "roi_global_context": roi_global_context,
            "init_source": str(init_source.resolve()) if init_source is not None else None,
            "train_tile_prob": train_tile_prob,
            "train_tile_min_scale": train_tile_min_scale,
            "train_tile_max_scale": train_tile_max_scale,
            "train_tile_min_visibility": train_tile_min_visibility,
            "train_tile_anchor_class_power": train_tile_anchor_class_power,
            "train_image_sampling_power": train_image_sampling_power,
            "roi_focal_gamma": roi_focal_gamma,
            "roi_class_weight_power": roi_class_weight_power,
            "roi_quality_loss_weight": roi_quality_weight,
            "roi_quality_jitter_count": roi_quality_jitter_count,
            "roi_quality_background_count": roi_quality_background_count,
            "roi_quality_inference_power": roi_quality_inference_power,
            "segmentation_support_power": segmentation_support_power,
            "segmentation_support_threshold": segmentation_support_threshold,
            "segmentation_box_refine": segmentation_box_refine,
            "segmentation_box_refine_threshold": segmentation_box_refine_threshold,
            "segmentation_box_refine_expansion": segmentation_box_refine_expansion,
            "segmentation_box_refine_blend": segmentation_box_refine_blend,
            "segmentation_box_refine_min_pixels": segmentation_box_refine_min_pixels,
            "count_aware_topk": count_aware_topk,
            "count_aware_multiplier": count_aware_multiplier,
            "count_aware_bias": count_aware_bias,
            "count_aware_min": count_aware_min,
            "count_aware_max": count_aware_max,
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
    base_meta["class_agnostic_detection"] = bool(class_agnostic_detection)
    base_meta["detection_classes"] = int(detection_classes)
    base_meta["decoupled_decoder"] = bool(decoupled_decoder)
    base_meta["decoder_branching"] = "det_cls_vs_seg_count" if decoupled_decoder else "shared"
    base_meta["dense_count_residual"] = bool(dense_count_residual)
    base_meta["task_aligned_detection"] = bool(task_aligned_detection)
    base_meta["highres_detection"] = bool(highres_detection)
    base_meta["roi_global_context"] = bool(roi_global_context)
    base_meta["train_tile_prob"] = float(train_tile_prob)
    base_meta["train_tile_min_scale"] = float(train_tile_min_scale)
    base_meta["train_tile_max_scale"] = float(train_tile_max_scale)
    base_meta["train_tile_min_visibility"] = float(train_tile_min_visibility)
    base_meta["train_tile_anchor_class_power"] = float(train_tile_anchor_class_power)
    base_meta["train_image_sampling_power"] = float(train_image_sampling_power)
    base_meta["roi_focal_gamma"] = float(roi_focal_gamma)
    base_meta["roi_class_weight_power"] = float(roi_class_weight_power)
    base_meta["roi_quality_loss_weight"] = float(roi_quality_weight)
    base_meta["roi_quality_jitter_count"] = int(roi_quality_jitter_count)
    base_meta["roi_quality_background_count"] = int(roi_quality_background_count)
    base_meta["roi_quality_inference_power"] = float(roi_quality_inference_power)
    base_meta["segmentation_support_power"] = float(segmentation_support_power)
    base_meta["segmentation_support_threshold"] = float(segmentation_support_threshold)
    base_meta["segmentation_box_refine"] = bool(segmentation_box_refine)
    base_meta["segmentation_box_refine_threshold"] = float(segmentation_box_refine_threshold)
    base_meta["segmentation_box_refine_expansion"] = float(segmentation_box_refine_expansion)
    base_meta["segmentation_box_refine_blend"] = float(segmentation_box_refine_blend)
    base_meta["segmentation_box_refine_min_pixels"] = int(segmentation_box_refine_min_pixels)
    base_meta["count_aware_topk"] = bool(count_aware_topk)
    base_meta["count_aware_multiplier"] = float(count_aware_multiplier)
    base_meta["count_aware_bias"] = float(count_aware_bias)
    base_meta["count_aware_min"] = int(count_aware_min)
    base_meta["count_aware_max"] = int(count_aware_max)
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
