from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .annotations import prepare_annotations
from .config import output_dirs
from .datasets import (
    CropClassificationDataset,
    CountingDataset,
    DetectionDataset,
    SegmentationDataset,
    detection_collate,
)
from .metrics import (
    SegmentationMeter,
    classification_metrics,
    classification_report_df,
    confusion_df,
    counting_metrics,
    detection_metrics,
)
from .models import create_detection_model, create_segmentation_model, create_timm_head_model
from .plots import save_confusion_matrix, save_history_plot, save_prediction_overlay
from .utils import json_dump, now_stamp, resolve_device, set_seed, worker_count


LOWER_IS_BETTER = {"mae", "rmse", "mape", "loss", "val_loss"}


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


def _split(frame: pd.DataFrame, split: str, limit: int | None = None) -> pd.DataFrame:
    output = frame[frame["split"] == split].copy()
    if limit is not None:
        output = output.sample(n=min(limit, len(output)), random_state=17)
    return output.reset_index(drop=True)


def _task_cfg(config: dict[str, Any], task: str, method: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("task_defaults", {}).get(task, {}))
    cfg.update(config.get("tasks", {}).get(task, {}).get(method, {}))
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    return cfg


def _is_better(metric_name: str, value: float, best: float | None) -> bool:
    if best is None:
        return True
    if metric_name in LOWER_IS_BETTER:
        return value < best
    return value > best


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(config.get("lr", 0.0003)),
        weight_decay=float(config.get("weight_decay", 0.05)),
    )


def _run_dir(config: dict[str, Any], task: str, method: str, seed: int) -> Path:
    dirs = output_dirs(config)
    run_dir = dirs[task] / "runs" / f"{now_stamp()}_{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")
    return run_dir


def _save_common_metadata(run_dir: Path, task: str, method: str, method_cfg: dict[str, Any], created, seed: int, device: torch.device) -> None:
    metadata = {
        "task": task,
        "method": method,
        "display_name": method_cfg.get("display_name", method),
        "family": method_cfg.get("family", "unknown"),
        "resolved_model_name": created.resolved_name,
        "seed": seed,
        "device": str(device),
        **created.metadata,
    }
    json_dump(metadata, run_dir / "metadata.json")


def _classification_epoch(model, loader, criterion, optimizer, device, amp: bool, train: bool) -> tuple[dict[str, float], pd.DataFrame]:
    model.train(train)
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    paths: list[str] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")

    for images, labels, batch_paths in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        probs = torch.softmax(logits.detach(), dim=1)
        conf, pred = probs.max(dim=1)
        running_loss += float(loss.item()) * labels.numel()
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        scores.extend(conf.cpu().tolist())
        paths.extend(list(batch_paths))

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    predictions = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred, "confidence": scores})
    return metrics, predictions


def _label_weights(labels: pd.Series, num_classes: int) -> torch.Tensor:
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).to_numpy(dtype=np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-9)
    weights = np.clip(weights, 0.25, 8.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def run_classification(
    config: dict[str, Any],
    method: str,
    seed: int,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    classes = list(config["classes"])
    method_cfg = _task_cfg(config, "classification", method, {"epochs": epochs, "batch_size": batch_size})
    training_cfg = {**config.get("training", {}), **method_cfg}
    if pretrained is not None:
        training_cfg["pretrained"] = pretrained
    set_seed(seed)
    device = resolve_device(device_name)
    crops = pd.read_csv(dirs["annotations"] / "classification_crops.csv")
    image_size = int(method_cfg.get("image_size", 224))
    loader_kwargs = _loader_kwargs(config, device)
    train_frame = _split(crops, "train", limit)
    train_dataset = CropClassificationDataset(train_frame, image_size, augment=True)
    sampler = None
    shuffle_train = True
    if bool(method_cfg.get("balanced_sampler", False)):
        class_weights = _label_weights(train_frame["label"], len(classes)).numpy()
        sample_weights = class_weights[train_frame["label"].to_numpy(dtype=int)]
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(method_cfg.get("batch_size", 24)),
        shuffle=shuffle_train,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        CropClassificationDataset(_split(crops, "val", limit), image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 24)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        CropClassificationDataset(_split(crops, "test", limit), image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 24)),
        shuffle=False,
        **loader_kwargs,
    )

    created = create_timm_head_model(method_cfg, num_outputs=len(classes), pretrained=bool(training_cfg.get("pretrained", True)))
    model = created.model.to(device)
    optimizer = _optimizer(model, training_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(method_cfg.get("epochs", 40))))
    loss_weight = None
    if bool(method_cfg.get("class_weighted_loss", False)):
        loss_weight = _label_weights(train_frame["label"], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=float(method_cfg.get("label_smoothing", 0.05)))
    run_dir = _run_dir(config, "classification", method, seed)
    _save_common_metadata(run_dir, "classification", method, method_cfg, created, seed, device)

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    val_metric = str(method_cfg.get("val_metric", "macro_f1"))
    amp = bool(training_cfg.get("amp", True))
    start = time.perf_counter()
    for epoch in range(1, int(method_cfg.get("epochs", 40)) + 1):
        train_metrics, _ = _classification_epoch(model, train_loader, criterion, optimizer, device, amp, train=True)
        val_metrics, _ = _classification_epoch(model, val_loader, criterion, optimizer, device, amp, train=False)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_history_plot(pd.DataFrame(history), run_dir / "training_curves.png", val_metric)
        score = float(val_metrics[val_metric])
        if _is_better(val_metric, score, best_value):
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_cfg.get("early_stopping_patience", 8)):
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, predictions = _classification_epoch(model, test_loader, criterion, optimizer, device, amp, train=False)
    predictions["true_class"] = predictions["y_true"].map(dict(enumerate(classes)))
    predictions["pred_class"] = predictions["y_pred"].map(dict(enumerate(classes)))
    predictions.to_csv(run_dir / "predictions_test.csv", index=False)
    report = classification_report_df(predictions["y_true"].tolist(), predictions["y_pred"].tolist(), classes)
    report.to_csv(run_dir / "classification_report_test.csv", index=False)
    matrix = confusion_df(predictions["y_true"].tolist(), predictions["y_pred"].tolist(), classes)
    matrix.to_csv(run_dir / "confusion_matrix_test.csv")
    save_confusion_matrix(matrix, run_dir / "confusion_matrix_test.png")
    test_metrics.update({"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": time.perf_counter() - start})
    json_dump(test_metrics, run_dir / "test_metrics.json")
    return run_dir


def _counting_epoch(model, loader, criterion, optimizer, device, amp: bool, train: bool) -> tuple[dict[str, float], pd.DataFrame]:
    model.train(train)
    running_loss = 0.0
    y_true: list[float] = []
    y_pred: list[float] = []
    paths: list[str] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")
    for images, targets, batch_paths in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            if output.ndim == 1:
                output = output[:, None]
            loss = criterion(output.float(), targets.float())
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += float(loss.item()) * targets.shape[0]
        y_true.extend(targets.detach().cpu().numpy().reshape(-1).tolist())
        y_pred.extend(output.detach().cpu().numpy().reshape(-1).tolist())
        paths.extend(list(batch_paths))
    metrics = counting_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    predictions = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred, "error": np.asarray(y_pred) - np.asarray(y_true)})
    return metrics, predictions


def run_counting(
    config: dict[str, Any],
    method: str,
    seed: int,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    method_cfg = _task_cfg(config, "counting", method, {"epochs": epochs, "batch_size": batch_size})
    training_cfg = {**config.get("training", {}), **method_cfg}
    if pretrained is not None:
        training_cfg["pretrained"] = pretrained
    set_seed(seed)
    device = resolve_device(device_name)
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    image_size = int(method_cfg.get("image_size", 384))
    loader_kwargs = _loader_kwargs(config, device)
    train_loader = DataLoader(
        CountingDataset(_split(images, "train", limit), image_size, str(method_cfg.get("target", "total")), augment=True),
        batch_size=int(method_cfg.get("batch_size", 12)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        CountingDataset(_split(images, "val", limit), image_size, str(method_cfg.get("target", "total")), augment=False),
        batch_size=int(method_cfg.get("batch_size", 12)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        CountingDataset(_split(images, "test", limit), image_size, str(method_cfg.get("target", "total")), augment=False),
        batch_size=int(method_cfg.get("batch_size", 12)),
        shuffle=False,
        **loader_kwargs,
    )

    created = create_timm_head_model(method_cfg, num_outputs=1, pretrained=bool(training_cfg.get("pretrained", True)))
    model = created.model.to(device)
    optimizer = _optimizer(model, training_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(method_cfg.get("epochs", 40))))
    criterion = nn.SmoothL1Loss()
    run_dir = _run_dir(config, "counting", method, seed)
    _save_common_metadata(run_dir, "counting", method, method_cfg, created, seed, device)

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    val_metric = str(method_cfg.get("val_metric", "mae"))
    amp = bool(training_cfg.get("amp", True))
    start = time.perf_counter()
    for epoch in range(1, int(method_cfg.get("epochs", 40)) + 1):
        train_metrics, _ = _counting_epoch(model, train_loader, criterion, optimizer, device, amp, train=True)
        val_metrics, _ = _counting_epoch(model, val_loader, criterion, optimizer, device, amp, train=False)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_history_plot(pd.DataFrame(history), run_dir / "training_curves.png", val_metric)
        score = float(val_metrics[val_metric])
        if _is_better(val_metric, score, best_value):
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_cfg.get("early_stopping_patience", 8)):
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, predictions = _counting_epoch(model, test_loader, criterion, optimizer, device, amp, train=False)
    predictions.to_csv(run_dir / "predictions_test.csv", index=False)
    test_metrics.update({"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": time.perf_counter() - start})
    json_dump(test_metrics, run_dir / "test_metrics.json")
    return run_dir


def _segmentation_epoch(model, loader, criterion, optimizer, device, amp: bool, train: bool, num_classes: int, class_names: list[str]):
    model.train(train)
    running_loss = 0.0
    meter = SegmentationMeter(num_classes)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and device.type == "cuda")
    sample_payload = None
    for images, masks, paths in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            logits = output["out"] if isinstance(output, dict) else output
            loss = criterion(logits, masks)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += float(loss.item()) * images.shape[0]
        meter.update(logits.detach(), masks.detach())
        if sample_payload is None and not train:
            sample_payload = (paths[0], masks[0].detach().cpu().numpy(), logits[0].detach().argmax(dim=0).cpu().numpy())
    metrics, per_class = meter.metrics(class_names)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics, per_class, sample_payload


def run_segmentation(
    config: dict[str, Any],
    method: str,
    seed: int,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    classes = list(config["classes"])
    method_cfg = _task_cfg(config, "segmentation", method, {"epochs": epochs, "batch_size": batch_size})
    training_cfg = {**config.get("training", {}), **method_cfg}
    if pretrained is not None:
        training_cfg["pretrained"] = pretrained
    set_seed(seed)
    device = resolve_device(device_name)
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    image_size = int(method_cfg.get("image_size", 512))
    loader_kwargs = _loader_kwargs(config, device)
    train_loader = DataLoader(
        SegmentationDataset(_split(images, "train", limit), image_size, augment=True),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        SegmentationDataset(_split(images, "val", limit), image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        SegmentationDataset(_split(images, "test", limit), image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )

    created = create_segmentation_model(method_cfg, num_classes=len(classes) + 1, pretrained=bool(training_cfg.get("pretrained", True)))
    model = created.model.to(device)
    optimizer = _optimizer(model, training_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(method_cfg.get("epochs", 60))))
    criterion = nn.CrossEntropyLoss()
    run_dir = _run_dir(config, "segmentation", method, seed)
    _save_common_metadata(run_dir, "segmentation", method, method_cfg, created, seed, device)

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    val_metric = str(method_cfg.get("val_metric", "miou_foreground"))
    amp = bool(training_cfg.get("amp", True))
    start = time.perf_counter()
    for epoch in range(1, int(method_cfg.get("epochs", 60)) + 1):
        train_metrics, _, _ = _segmentation_epoch(model, train_loader, criterion, optimizer, device, amp, True, len(classes) + 1, classes)
        val_metrics, val_per_class, _ = _segmentation_epoch(model, val_loader, criterion, optimizer, device, amp, False, len(classes) + 1, classes)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        val_per_class.to_csv(run_dir / "per_class_val.csv", index=False)
        save_history_plot(pd.DataFrame(history), run_dir / "training_curves.png", val_metric)
        score = float(val_metrics[val_metric])
        if _is_better(val_metric, score, best_value):
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_cfg.get("early_stopping_patience", 8)):
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, per_class, sample = _segmentation_epoch(model, test_loader, criterion, optimizer, device, amp, False, len(classes) + 1, classes)
    per_class.to_csv(run_dir / "per_class_test.csv", index=False)
    if sample is not None:
        image_path, target, pred = sample
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, target, pred, run_dir / "sample_prediction_overlay.jpg")
    test_metrics.update({"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": time.perf_counter() - start})
    json_dump(test_metrics, run_dir / "test_metrics.json")
    return run_dir


def _targets_to_numpy(targets: list[dict[str, torch.Tensor]]) -> list[dict[str, np.ndarray]]:
    output = []
    for target in targets:
        output.append(
            {
                "boxes": target["boxes"].detach().cpu().numpy(),
                "labels": target["labels"].detach().cpu().numpy(),
            }
        )
    return output


def _predictions_to_numpy(predictions: list[dict[str, torch.Tensor]], score_threshold: float) -> list[dict[str, np.ndarray]]:
    output = []
    for pred in predictions:
        scores = pred["scores"].detach().cpu().numpy()
        keep = scores >= score_threshold
        output.append(
            {
                "boxes": pred["boxes"].detach().cpu().numpy()[keep],
                "labels": pred["labels"].detach().cpu().numpy()[keep],
                "scores": scores[keep],
            }
        )
    return output


def _detection_train_epoch(model, loader, optimizer, device, amp: bool) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    totals: dict[str, float] = {}
    count = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(images)
        count += batch_size
        totals["loss"] = totals.get("loss", 0.0) + float(loss.item()) * batch_size
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.inference_mode()
def _detection_eval(model, loader, device, num_classes: int, score_threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    all_predictions: list[dict[str, np.ndarray]] = []
    all_targets: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = [image.to(device, non_blocking=True) for image in images]
        predictions = model(images)
        pred_np = _predictions_to_numpy(predictions, score_threshold)
        target_np = _targets_to_numpy(targets)
        all_predictions.extend(pred_np)
        all_targets.extend(target_np)
        for image_offset, pred in enumerate(pred_np):
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                rows.append(
                    {
                        "batch_image_index": len(all_targets) - len(target_np) + image_offset,
                        "label": int(label),
                        "score": float(score),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    }
                )
    metrics, per_class = detection_metrics(all_predictions, all_targets, num_classes=num_classes)
    predictions_df = pd.DataFrame(rows)
    predictions_df.attrs["per_class"] = per_class
    return metrics, predictions_df


def run_detection(
    config: dict[str, Any],
    method: str,
    seed: int,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    classes = list(config["classes"])
    method_cfg = _task_cfg(config, "detection", method, {"epochs": epochs, "batch_size": batch_size})
    training_cfg = {**config.get("training", {}), **method_cfg}
    if pretrained is not None:
        training_cfg["pretrained"] = pretrained
    set_seed(seed)
    device = resolve_device(device_name)
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    image_size = int(method_cfg.get("image_size", 768))
    include_masks = str(method_cfg.get("model_name", "")).startswith("maskrcnn")
    loader_kwargs = _loader_kwargs(config, device)
    train_images = _split(images, "train", limit)
    val_images = _split(images, "val", limit)
    test_images = _split(images, "test", limit)
    train_instances = instances[instances["stem"].isin(set(train_images["stem"]))]
    val_instances = instances[instances["stem"].isin(set(val_images["stem"]))]
    test_instances = instances[instances["stem"].isin(set(test_images["stem"]))]
    train_loader = DataLoader(
        DetectionDataset(train_images, train_instances, image_size, include_masks, augment=True),
        batch_size=int(method_cfg.get("batch_size", 2)),
        shuffle=True,
        collate_fn=detection_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        DetectionDataset(val_images, val_instances, image_size, include_masks, augment=False),
        batch_size=int(method_cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=detection_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        DetectionDataset(test_images, test_instances, image_size, include_masks, augment=False),
        batch_size=int(method_cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=detection_collate,
        **loader_kwargs,
    )

    created = create_detection_model(method_cfg, num_classes_with_background=len(classes) + 1, pretrained=bool(training_cfg.get("pretrained", True)))
    model = created.model.to(device)
    optimizer = _optimizer(model, training_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(method_cfg.get("epochs", 50))))
    run_dir = _run_dir(config, "detection", method, seed)
    _save_common_metadata(run_dir, "detection", method, method_cfg, created, seed, device)

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    val_metric = str(method_cfg.get("val_metric", "map50"))
    amp = bool(training_cfg.get("amp", True))
    score_threshold = float(method_cfg.get("score_threshold", 0.25))
    start = time.perf_counter()
    for epoch in range(1, int(method_cfg.get("epochs", 50)) + 1):
        train_metrics = _detection_train_epoch(model, train_loader, optimizer, device, amp)
        val_metrics, val_predictions = _detection_eval(model, val_loader, device, len(classes), score_threshold)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_history_plot(pd.DataFrame(history), run_dir / "training_curves.png", val_metric)
        per_class = val_predictions.attrs.get("per_class")
        if per_class is not None:
            per_class.to_csv(run_dir / "per_class_val.csv", index=False)
        score = float(val_metrics[val_metric])
        if _is_better(val_metric, score, best_value):
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_cfg.get("early_stopping_patience", 8)):
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, predictions = _detection_eval(model, test_loader, device, len(classes), score_threshold)
    predictions.to_csv(run_dir / "predictions_test.csv", index=False)
    per_class = predictions.attrs.get("per_class")
    if per_class is not None:
        per_class["class_name"] = per_class["class_id"].map({idx + 1: name for idx, name in enumerate(classes)})
        per_class.to_csv(run_dir / "per_class_test.csv", index=False)
    test_metrics.update({"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": time.perf_counter() - start})
    json_dump(test_metrics, run_dir / "test_metrics.json")
    return run_dir


def run_task(
    config: dict[str, Any],
    task: str,
    method: str,
    seed: int,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
) -> Path:
    runners = {
        "classification": run_classification,
        "counting": run_counting,
        "segmentation": run_segmentation,
        "detection": run_detection,
    }
    if task not in runners:
        raise ValueError(f"Unknown task {task!r}. Expected one of {sorted(runners)}")
    if method not in config.get("tasks", {}).get(task, {}):
        raise ValueError(f"Unknown method {method!r} for task {task!r}.")
    return runners[task](
        config=config,
        method=method,
        seed=seed,
        device_name=device_name,
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        pretrained=pretrained,
    )


def run_all(
    config: dict[str, Any],
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int | None = None,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    pretrained: bool | None = None,
    stop_on_error: bool = False,
) -> dict[str, str]:
    failures: dict[str, str] = {}
    tasks_to_run = tasks or list(config.get("tasks", {}).keys())
    seed = int(seed if seed is not None else config.get("training", {}).get("seed", 42))
    prepare_annotations(config)
    for task in tasks_to_run:
        task_methods = list(config.get("tasks", {}).get(task, {}).keys())
        if methods:
            task_methods = [method for method in task_methods if method in methods]
        for method in task_methods:
            key = f"{task}/{method}"
            try:
                print(f"\n=== {key} seed={seed} ===")
                run_task(config, task, method, seed, device_name, epochs, batch_size, limit, pretrained)
            except Exception as exc:
                failures[key] = str(exc)
                print(f"FAILED {key}: {exc}")
                if stop_on_error:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return failures
