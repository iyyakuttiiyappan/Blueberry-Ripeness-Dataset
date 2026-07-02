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
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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
from .plots import save_confusion_matrix, save_history_plot, save_prediction_overlay
from .utils import json_dump, now_stamp, resolve_device, set_seed, worker_count


OURS_METHOD = "berrymtl_unified"
OURS_DISPLAY = "BerryMTL-ConvNeXtV2 (ours)"


class BerryMTLDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
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
        image_tensor = VF.normalize(VF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)
        count = torch.tensor([float(row["Total"])], dtype=torch.float32)
        return image_tensor, mask_tensor, count, str(row["stem"])


class BerryMTLNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        decoder_channels: int = 128,
    ):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels])
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels * len(channels), decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        features = [self._nchw(feature, channel) for feature, channel in zip(features, self.encoder.feature_info.channels())]
        target_size = features[0].shape[-2:]
        projected = []
        for feature, conv in zip(features, self.lateral):
            projected.append(F.interpolate(conv(feature), size=target_size, mode="bilinear", align_corners=False))
        logits = self.seg_head(torch.cat(projected, dim=1))
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        count = self.count_head(features[-1])
        return {"seg": logits, "count": count}


def _split(frame: pd.DataFrame, split: str, limit: int | None = None) -> pd.DataFrame:
    output = frame[frame["split"] == split].copy()
    if limit is not None:
        output = output.sample(n=min(limit, len(output)), random_state=17)
    return output.reset_index(drop=True)


def _loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = worker_count(int(config.get("training", {}).get("num_workers", 0)))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def _class_weights(frame: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for path in frame["semantic_mask_path"]:
        mask = np.asarray(_mask(path), dtype=np.int64).reshape(-1)
        counts += np.bincount(mask, minlength=num_classes)
    freq = counts / max(1.0, counts.sum())
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    weights[0] *= 0.35
    weights = np.clip(weights, 0.2, 8.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(min=0, max=num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    probs = probs[:, 1:]
    target_oh = target_oh[:, 1:]
    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dims)
    denom = probs.sum(dims) + target_oh.sum(dims)
    dice = (2 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def _joint_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    count_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    train: bool,
    num_classes: int,
    class_names: list[str],
    count_weight: float,
    dice_weight: float,
) -> dict[str, float]:
    model.train(train)
    meter = SegmentationMeter(num_classes)
    y_true: list[float] = []
    y_pred: list[float] = []
    total_loss = 0.0
    seg_loss_total = 0.0
    count_loss_total = 0.0
    count = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")
    for images, masks, targets, _ in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            logits = output["seg"]
            counts = output["count"].float()
            seg_loss = ce_loss(logits, masks) + dice_weight * _dice_loss(logits, masks, num_classes)
            c_loss = count_loss(counts, targets.float())
            loss = seg_loss + count_weight * c_loss
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch = images.shape[0]
        count += batch
        total_loss += float(loss.item()) * batch
        seg_loss_total += float(seg_loss.item()) * batch
        count_loss_total += float(c_loss.item()) * batch
        meter.update(logits.detach(), masks.detach())
        y_true.extend(targets.detach().cpu().numpy().reshape(-1).tolist())
        y_pred.extend(counts.detach().cpu().numpy().reshape(-1).tolist())
    seg_metrics, _ = meter.metrics(class_names)
    count_metrics = counting_metrics(np.asarray(y_true), np.asarray(y_pred))
    output = {
        "loss": total_loss / max(1, count),
        "seg_loss": seg_loss_total / max(1, count),
        "count_loss": count_loss_total / max(1, count),
        **{f"seg_{key}": value for key, value in seg_metrics.items()},
        **{f"count_{key}": value for key, value in count_metrics.items()},
        "joint_score": float(seg_metrics["miou_foreground"] - 0.002 * count_metrics["mae"]),
    }
    return output


def _gt_by_image(
    image_frame: pd.DataFrame,
    instances: pd.DataFrame,
    image_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=False)}
    targets: dict[str, dict[str, np.ndarray]] = {}
    for row in image_frame.itertuples(index=False):
        group = grouped.get(row.stem, pd.DataFrame())
        if group.empty:
            targets[str(row.stem)] = {"boxes": np.zeros((0, 4), dtype=float), "labels": np.zeros((0,), dtype=int)}
            continue
        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).copy()
        boxes[:, [0, 2]] *= image_size / float(row.aligned_width)
        boxes[:, [1, 3]] *= image_size / float(row.aligned_height)
        targets[str(row.stem)] = {"boxes": boxes, "labels": group["det_label"].to_numpy(dtype=int)}
    return targets


def _components_from_prediction(
    pred_mask: np.ndarray,
    probs: np.ndarray,
    min_area: int,
) -> dict[str, np.ndarray]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []
    for class_id in range(1, probs.shape[0]):
        labeled, _ = ndimage.label(pred_mask == class_id)
        for component_slice in ndimage.find_objects(labeled):
            if component_slice is None:
                continue
            component = labeled[component_slice] > 0
            area = int(component.sum())
            if area < min_area:
                continue
            ys, xs = component_slice
            y1, y2 = int(ys.start), int(ys.stop)
            x1, x2 = int(xs.start), int(xs.stop)
            score = float(probs[class_id, ys, xs][component].mean()) if area > 0 else 0.0
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            scores.append(score)
    return {
        "boxes": np.asarray(boxes, dtype=float).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=int),
        "scores": np.asarray(scores, dtype=float),
    }


def _classify_instances_from_probs(
    probs: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    height, width = probs.shape[1:]
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
        region = probs[1:, y1:y2, x1:x2]
        if region.size == 0:
            avg = np.zeros((len(class_names),), dtype=float)
        else:
            avg = region.reshape(len(class_names), -1).mean(axis=1)
        pred = int(np.argmax(avg))
        rows.append(
            {
                "instance_index": idx,
                "y_true": int(label) - 1,
                "y_pred": pred,
                "confidence": float(avg[pred]) if len(avg) else 0.0,
                "true_class": class_names[int(label) - 1],
                "pred_class": class_names[pred],
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def _evaluate_ours(
    model: nn.Module,
    loader: DataLoader,
    image_frame: pd.DataFrame,
    instances: pd.DataFrame,
    device: torch.device,
    amp: bool,
    image_size: int,
    class_names: list[str],
    min_component_area: int,
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
    for images, masks, targets, stems in tqdm(loader, desc="test", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            logits = output["seg"]
            counts = output["count"].float()
        probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_batch = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        true_batch = masks.detach().cpu().numpy().astype(np.uint8)
        seg_meter.update(logits, masks)
        count_true.extend(targets.numpy().reshape(-1).tolist())
        count_pred.extend(counts.detach().cpu().numpy().reshape(-1).tolist())
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
            cls = _classify_instances_from_probs(probs_batch[local_idx], target_det["boxes"], target_det["labels"], class_names)
            cls["stem"] = stem
            cls["filename"] = row["filename"]
            cls["path"] = row["image_path"]
            classification_rows.append(cls)
            if sample_payload is None:
                sample_payload = (row["image_path"], true_mask, pred_mask)

    seg_metrics, per_class_seg = seg_meter.metrics(class_names)
    count_metrics = counting_metrics(np.asarray(count_true), np.asarray(count_pred))
    detection_metric_values, per_class_det = detection_metrics(detection_predictions, detection_targets, len(class_names))
    cls_predictions = pd.concat(classification_rows, ignore_index=True) if classification_rows else pd.DataFrame()
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


def _metadata(task: str, seed: int, device: torch.device, image_size: int, model_name: str, parameters: int) -> dict[str, Any]:
    return {
        "task": task,
        "method": OURS_METHOD,
        "display_name": OURS_DISPLAY,
        "family": "Unified multitask dense prediction",
        "resolved_model_name": f"timm_fpn_multitask:{model_name}",
        "seed": seed,
        "device": str(device),
        "parameters": int(parameters),
        "trainable_parameters": int(parameters),
        "image_size": int(image_size),
        "shared_backbone": True,
    }


def _mirror_task_outputs(
    config: dict[str, Any],
    shared_dir: Path,
    task: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    files: dict[str, Path],
) -> Path:
    dirs = output_dirs(config)
    run_dir = dirs[task] / "runs" / shared_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(metrics, run_dir / "test_metrics.json")
    json_dump(metadata, run_dir / "metadata.json")
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")
    for target_name, source in files.items():
        if source.exists():
            shutil.copy2(source, run_dir / target_name)
    for common in ["history.csv", "training_curves.png"]:
        source = shared_dir / common
        if source.exists():
            shutil.copy2(source, run_dir / common)
    return run_dir


def run_berrymtl_unified(
    config: dict[str, Any],
    seed: int | None = None,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    seed = int(seed if seed is not None else config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(device_name)
    classes = list(config["classes"])
    num_classes = len(classes) + 1
    ours_cfg = config.get("ours", {})
    model_name = str(ours_cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))
    image_size = int(ours_cfg.get("image_size", 512))
    batch_size = int(batch_size if batch_size is not None else ours_cfg.get("batch_size", 4))
    epochs = int(epochs if epochs is not None else ours_cfg.get("epochs", 60))
    count_weight = float(ours_cfg.get("count_loss_weight", 0.04))
    dice_weight = float(ours_cfg.get("dice_loss_weight", 0.6))
    min_component_area = int(ours_cfg.get("min_component_area", 8))
    pretrained = bool(config.get("training", {}).get("pretrained", True) if pretrained is None else pretrained)

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    train_images = _split(images, "train", limit)
    val_images = _split(images, "val", limit)
    test_images = _split(images, "test", limit)
    test_instances = instances[instances["stem"].isin(set(test_images["stem"]))]
    loader_kwargs = _loader_kwargs(config, device)
    train_loader = DataLoader(BerryMTLDataset(train_images, image_size, True), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(BerryMTLDataset(val_images, image_size, False), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(BerryMTLDataset(test_images, image_size, False), batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = BerryMTLNet(model_name, num_classes=num_classes, pretrained=pretrained).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(ours_cfg.get("lr", config.get("training", {}).get("lr", 0.0003))),
        weight_decay=float(ours_cfg.get("weight_decay", config.get("training", {}).get("weight_decay", 0.05))),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    weights = _class_weights(train_images, num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    count_loss = nn.SmoothL1Loss()
    amp = bool(config.get("training", {}).get("amp", True))
    patience = int(ours_cfg.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 8)))
    run_dir = dirs["analysis"] / "ours" / "runs" / f"{now_stamp()}_{OURS_METHOD}_seed{seed}"
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
        train_metrics = _joint_epoch(
            model, train_loader, ce_loss, count_loss, optimizer, device, amp, True, num_classes, classes, count_weight, dice_weight
        )
        val_metrics = _joint_epoch(
            model, val_loader, ce_loss, count_loss, optimizer, device, amp, False, num_classes, classes, count_weight, dice_weight
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
    test = _evaluate_ours(
        model,
        test_loader,
        test_images,
        test_instances,
        device,
        amp,
        image_size,
        classes,
        min_component_area=min_component_area,
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
    save_confusion_matrix(test["classification_confusion"], run_dir / "confusion_matrix_test.png", title=f"{OURS_DISPLAY}: ROI Classification")
    if test["sample"] is not None:
        image_path, target, pred = test["sample"]
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, target, pred, run_dir / "sample_prediction_overlay.jpg")

    common = {"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": train_seconds}
    json_dump({"method": OURS_METHOD, "display_name": OURS_DISPLAY, "best_epoch": best_epoch, "best_val_metric": best_value}, run_dir / "metadata.json")
    combined = {
        "classification": {**test["classification"], **common},
        "counting": {**test["counting"], **common},
        "segmentation": {**test["segmentation"], **common},
        "detection": {**test["detection"], **common},
    }
    json_dump(combined, run_dir / "test_metrics_by_task.json")

    base_meta = _metadata("multitask", seed, device, image_size, model_name, parameters)
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
