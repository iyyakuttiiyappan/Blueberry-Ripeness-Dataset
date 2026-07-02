from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
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
    BerryMTLDataset,
    BerryMTLNet,
    _gt_by_image,
    _loader_kwargs,
    _metadata,
    _mirror_task_outputs,
    _split,
)
from .plots import save_confusion_matrix, save_prediction_overlay
from .utils import json_dump, json_load, now_stamp, resolve_device, set_seed


REFINED_METHOD = "berrymtl_refined"
REFINED_DISPLAY = "BerryMTL-Refined (ours)"


def _latest_unified_run(config: dict[str, Any]) -> Path:
    dirs = output_dirs(config)
    run_root = dirs["analysis"] / "ours" / "runs"
    candidates = sorted(run_root.glob("*_berrymtl_unified_seed*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No BerryMTL unified runs found under {run_root}.")
    return candidates[0]


def _train_instance_stats(
    train_images: pd.DataFrame,
    train_instances: pd.DataFrame,
    image_size: int,
    num_classes: int,
) -> tuple[dict[int, float], np.ndarray]:
    image_shape = {
        str(row.stem): (float(row.aligned_width), float(row.aligned_height))
        for row in train_images.itertuples(index=False)
    }
    area_by_class: dict[int, list[float]] = {class_id: [] for class_id in range(1, num_classes + 1)}
    counts = np.zeros((num_classes,), dtype=np.float64)
    for row in train_instances.itertuples(index=False):
        width, height = image_shape.get(str(row.stem), (3000.0, 4000.0))
        scaled_area = float(row.area) * (image_size / width) * (image_size / height)
        class_id = int(row.det_label)
        area_by_class.setdefault(class_id, []).append(max(4.0, scaled_area))
        counts[class_id - 1] += 1.0

    all_areas = [value for values in area_by_class.values() for value in values]
    fallback = float(np.median(all_areas)) if all_areas else 96.0
    median_area = {
        class_id: float(np.median(values)) if values else fallback
        for class_id, values in area_by_class.items()
    }

    prior_weights = 1.0 / np.sqrt(counts + 1.0)
    prior_weights = prior_weights / max(float(prior_weights.mean()), 1e-9)
    prior_weights = np.clip(prior_weights, 0.25, 4.0)
    return median_area, prior_weights.astype(np.float32)


def _peak_coords(
    distance: np.ndarray,
    max_peaks: int,
    min_distance: int,
    rel_threshold: float,
) -> np.ndarray:
    if max_peaks <= 1 or distance.size == 0:
        return np.zeros((0, 2), dtype=int)
    peak_floor = max(1.5, float(distance.max()) * float(rel_threshold))
    max_filtered = ndimage.maximum_filter(distance, size=max(3, 2 * int(min_distance) + 1), mode="constant")
    mask = (distance >= peak_floor) & (distance == max_filtered)
    coords = np.column_stack(np.nonzero(mask))
    if len(coords) == 0:
        return coords.astype(int)

    order = np.argsort(distance[coords[:, 0], coords[:, 1]])[::-1]
    selected: list[np.ndarray] = []
    min_sq = float(min_distance * min_distance)
    for idx in order:
        coord = coords[idx]
        if all(float(np.sum((coord - prior) ** 2)) >= min_sq for prior in selected):
            selected.append(coord)
        if len(selected) >= max_peaks:
            break
    if not selected:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(selected, dtype=int)


def _split_component(
    component: np.ndarray,
    expected_count: int,
    peak_distance: int,
    peak_rel_threshold: float,
    min_cluster_area: int,
) -> list[np.ndarray]:
    area = int(component.sum())
    if expected_count <= 1 or area < max(min_cluster_area * 2, 12):
        return [component]

    distance = ndimage.distance_transform_edt(component)
    peaks = _peak_coords(distance, expected_count, peak_distance, peak_rel_threshold)
    if len(peaks) <= 1:
        return [component]

    pixels = np.column_stack(np.nonzero(component))
    tree = cKDTree(peaks.astype(np.float32))
    _, assignments = tree.query(pixels.astype(np.float32), k=1)

    clusters: list[np.ndarray] = []
    for peak_idx in range(len(peaks)):
        selected = pixels[assignments == peak_idx]
        if len(selected) < min_cluster_area:
            continue
        cluster = np.zeros_like(component, dtype=bool)
        cluster[selected[:, 0], selected[:, 1]] = True
        clusters.append(cluster)
    return clusters if clusters else [component]


def _components_from_prediction_refined(
    pred_mask: np.ndarray,
    probs: np.ndarray,
    median_area: dict[int, float],
    min_component_area: int,
    prob_threshold: float,
    area_factor: float,
    peak_distance: int,
    peak_rel_threshold: float,
    min_cluster_area: int,
    max_splits: int,
) -> dict[str, np.ndarray]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []
    for class_id in range(1, probs.shape[0]):
        class_binary = pred_mask == class_id
        if prob_threshold > 0:
            class_binary &= probs[class_id] >= prob_threshold
        labeled, count = ndimage.label(class_binary)
        for component_id, component_slice in enumerate(ndimage.find_objects(labeled), start=1):
            if component_slice is None or component_id > count:
                continue
            component = labeled[component_slice] == component_id
            area = int(component.sum())
            if area < min_component_area:
                continue

            target_area = max(1.0, float(median_area.get(class_id, np.median(list(median_area.values())))))
            expected = int(round(area / max(1.0, target_area * area_factor)))
            expected = max(1, min(int(max_splits), expected))
            clusters = _split_component(component, expected, peak_distance, peak_rel_threshold, min_cluster_area)

            ys, xs = component_slice
            class_probs = probs[class_id, ys, xs]
            for cluster in clusters:
                cluster_area = int(cluster.sum())
                if cluster_area < min_cluster_area:
                    continue
                cy, cx = np.nonzero(cluster)
                y1 = int(ys.start + cy.min())
                y2 = int(ys.start + cy.max() + 1)
                x1 = int(xs.start + cx.min())
                x2 = int(xs.start + cx.max() + 1)
                values = class_probs[cluster]
                score = float(0.75 * values.mean() + 0.25 * values.max()) if values.size else 0.0
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
                scores.append(score)
    return {
        "boxes": np.asarray(boxes, dtype=float).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=int),
        "scores": np.asarray(scores, dtype=float),
    }


def _classify_instances_refined(
    probs: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    foreground_threshold: float,
    top_fraction: float,
    prior_weights: np.ndarray,
    prior_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    height, width = probs.shape[1:]
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
        region = probs[1:, y1:y2, x1:x2].astype(np.float32)
        if region.size == 0:
            avg = np.zeros((len(class_names),), dtype=np.float32)
        else:
            foreground = region.sum(axis=0)
            flat = foreground.reshape(-1)
            if top_fraction < 1.0 and flat.size > 1:
                keep = max(1, int(round(flat.size * float(top_fraction))))
                threshold = float(np.partition(flat, -keep)[-keep])
                mask = foreground >= max(float(foreground_threshold), threshold)
            else:
                mask = foreground >= float(foreground_threshold)
            if int(mask.sum()) < 3:
                threshold = float(np.quantile(flat, 0.65)) if flat.size else 0.0
                mask = foreground >= threshold
            if int(mask.sum()) < 1:
                avg = region.reshape(len(class_names), -1).mean(axis=1)
            else:
                avg = region[:, mask].mean(axis=1)

        adjusted = avg * np.power(prior_weights, float(prior_alpha))
        pred = int(np.argmax(adjusted)) if len(adjusted) else 0
        rows.append(
            {
                "instance_index": idx,
                "y_true": int(label) - 1,
                "y_pred": pred,
                "confidence": float(adjusted[pred]) if len(adjusted) else 0.0,
                "true_class": class_names[int(label) - 1],
                "pred_class": class_names[pred],
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def _collect_records(
    model: torch.nn.Module,
    loader: DataLoader,
    image_frame: pd.DataFrame,
    instances: pd.DataFrame,
    device: torch.device,
    amp: bool,
    image_size: int,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names) + 1
    model.eval()
    image_by_stem = image_frame.set_index("stem")
    instance_targets = _gt_by_image(image_frame, instances, image_size)
    seg_meter = SegmentationMeter(num_classes)
    count_true: list[float] = []
    count_pred: list[float] = []
    records: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []

    for images, masks, targets, stems in tqdm(loader, desc="refine inference", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            logits = output["seg"]
            counts = output["count"].float()
        probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float16)
        pred_batch = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        true_batch = masks.detach().cpu().numpy().astype(np.uint8)
        seg_meter.update(logits, masks)
        batch_true = targets.numpy().reshape(-1).tolist()
        batch_pred = counts.detach().cpu().numpy().reshape(-1).tolist()
        count_true.extend(batch_true)
        count_pred.extend(batch_pred)
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
            records.append(
                {
                    "stem": stem,
                    "filename": row["filename"],
                    "path": row["image_path"],
                    "probs": probs_batch[local_idx],
                    "pred_mask": pred_mask,
                    "true_mask": true_mask,
                    "target_det": instance_targets[stem],
                    "count_true": float(batch_true[local_idx]),
                    "count_pred": float(batch_pred[local_idx]),
                }
            )

    seg_metrics, per_class_seg = seg_meter.metrics(class_names)
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
        "records": records,
        "segmentation": seg_metrics,
        "segmentation_per_class": per_class_seg,
        "segmentation_per_image": pd.DataFrame(segmentation_rows),
        "counting": counting_metrics(np.asarray(count_true), np.asarray(count_pred)),
        "counting_predictions": count_predictions,
    }


def _detection_param_grid() -> list[dict[str, Any]]:
    rows = []
    for prob_threshold in [0.0, 0.12, 0.22]:
        for area_factor in [0.65, 0.85, 1.05, 1.25]:
            for peak_distance in [5, 7, 9]:
                rows.append(
                    {
                        "min_component_area": 6,
                        "prob_threshold": prob_threshold,
                        "area_factor": area_factor,
                        "peak_distance": peak_distance,
                        "peak_rel_threshold": 0.25,
                        "min_cluster_area": 5,
                        "max_splits": 24,
                    }
                )
    return rows


def _classification_param_grid() -> list[dict[str, Any]]:
    rows = []
    for foreground_threshold in [0.0, 0.15, 0.3]:
        for top_fraction in [0.25, 0.4, 0.6, 1.0]:
            for prior_alpha in [0.0, 0.2, 0.4, 0.6]:
                rows.append(
                    {
                        "foreground_threshold": foreground_threshold,
                        "top_fraction": top_fraction,
                        "prior_alpha": prior_alpha,
                    }
                )
    return rows


def _evaluate_detection_records(
    records: list[dict[str, Any]],
    class_names: list[str],
    median_area: dict[int, float],
    params: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    predictions = []
    targets = []
    prediction_rows: list[dict[str, Any]] = []
    for image_index, record in enumerate(records):
        pred = _components_from_prediction_refined(
            record["pred_mask"],
            record["probs"].astype(np.float32),
            median_area=median_area,
            min_component_area=int(params["min_component_area"]),
            prob_threshold=float(params["prob_threshold"]),
            area_factor=float(params["area_factor"]),
            peak_distance=int(params["peak_distance"]),
            peak_rel_threshold=float(params["peak_rel_threshold"]),
            min_cluster_area=int(params["min_cluster_area"]),
            max_splits=int(params["max_splits"]),
        )
        predictions.append(pred)
        targets.append(record["target_det"])
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            prediction_rows.append(
                {
                    "batch_image_index": image_index,
                    "label": int(label),
                    "score": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
    metrics, per_class = detection_metrics(predictions, targets, len(class_names))
    return metrics, per_class, pd.DataFrame(prediction_rows)


def _evaluate_classification_records(
    records: list[dict[str, Any]],
    class_names: list[str],
    prior_weights: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for record in records:
        target = record["target_det"]
        frame = _classify_instances_refined(
            record["probs"].astype(np.float32),
            target["boxes"],
            target["labels"],
            class_names,
            foreground_threshold=float(params["foreground_threshold"]),
            top_fraction=float(params["top_fraction"]),
            prior_weights=prior_weights,
            prior_alpha=float(params["prior_alpha"]),
        )
        frame["stem"] = record["stem"]
        frame["filename"] = record["filename"]
        frame["path"] = record["path"]
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics = classification_metrics(predictions["y_true"].tolist(), predictions["y_pred"].tolist())
    report = classification_report_df(predictions["y_true"].tolist(), predictions["y_pred"].tolist(), class_names)
    confusion = confusion_df(predictions["y_true"].tolist(), predictions["y_pred"].tolist(), class_names)
    return metrics, predictions, report, confusion


def _load_source_common(source_run: Path) -> dict[str, float]:
    path = source_run / "test_metrics_by_task.json"
    if not path.exists():
        return {}
    payload = json_load(path)
    values = next(iter(payload.values())) if isinstance(payload, dict) and payload else {}
    return {
        key: float(values[key])
        for key in ["best_epoch", "best_val_metric", "train_seconds"]
        if key in values and values[key] is not None
    }


def refine_berrymtl_unified(
    config: dict[str, Any],
    source_run: str | Path | None = None,
    seed: int | None = None,
    device_name: str | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
) -> Path:
    prepare_annotations(config)
    dirs = output_dirs(config)
    source = Path(source_run) if source_run is not None else _latest_unified_run(config)
    checkpoint_path = source / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {checkpoint_path}")

    seed = int(seed if seed is not None else config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(device_name)
    classes = list(config["classes"])
    num_classes = len(classes) + 1
    ours_cfg = config.get("ours", {})
    model_name = str(ours_cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))
    image_size = int(ours_cfg.get("image_size", 512))
    batch_size = int(batch_size if batch_size is not None else ours_cfg.get("batch_size", 4))
    amp = bool(config.get("training", {}).get("amp", True))

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    train_images = _split(images, "train", limit)
    val_images = _split(images, "val", limit)
    test_images = _split(images, "test", limit)
    train_instances = instances[instances["stem"].isin(set(train_images["stem"]))]
    val_instances = instances[instances["stem"].isin(set(val_images["stem"]))]
    test_instances = instances[instances["stem"].isin(set(test_images["stem"]))]

    loader_kwargs = _loader_kwargs(config, device)
    val_loader = DataLoader(BerryMTLDataset(val_images, image_size, False), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(BerryMTLDataset(test_images, image_size, False), batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = BerryMTLNet(model_name, num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())

    start = time.perf_counter()
    median_area, prior_weights = _train_instance_stats(train_images, train_instances, image_size, len(classes))
    val = _collect_records(model, val_loader, val_images, val_instances, device, amp, image_size, classes)
    test = _collect_records(model, test_loader, test_images, test_instances, device, amp, image_size, classes)

    detection_sweep_rows = []
    best_det_params: dict[str, Any] | None = None
    best_det_metrics: dict[str, float] | None = None
    for params in tqdm(_detection_param_grid(), desc="detection refinement", leave=False):
        metrics, _, _ = _evaluate_detection_records(val["records"], classes, median_area, params)
        row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
        detection_sweep_rows.append(row)
        score = (float(metrics["map50"]), float(metrics["map50_95"]), float(metrics["f1_50"]))
        best_score = (
            float(best_det_metrics["map50"]),
            float(best_det_metrics["map50_95"]),
            float(best_det_metrics["f1_50"]),
        ) if best_det_metrics is not None else (-1.0, -1.0, -1.0)
        if score > best_score:
            best_det_params = dict(params)
            best_det_metrics = dict(metrics)

    classification_sweep_rows = []
    best_cls_params: dict[str, Any] | None = None
    best_cls_metrics: dict[str, float] | None = None
    for params in tqdm(_classification_param_grid(), desc="classification refinement", leave=False):
        metrics, _, _, _ = _evaluate_classification_records(val["records"], classes, prior_weights, params)
        row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
        classification_sweep_rows.append(row)
        score = (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]), float(metrics["accuracy"]))
        best_score = (
            float(best_cls_metrics["macro_f1"]),
            float(best_cls_metrics["balanced_accuracy"]),
            float(best_cls_metrics["accuracy"]),
        ) if best_cls_metrics is not None else (-1.0, -1.0, -1.0)
        if score > best_score:
            best_cls_params = dict(params)
            best_cls_metrics = dict(metrics)

    if best_det_params is None or best_cls_params is None:
        raise RuntimeError("Refinement parameter search did not produce a candidate.")

    detection, det_per_class, det_predictions = _evaluate_detection_records(test["records"], classes, median_area, best_det_params)
    classification, cls_predictions, cls_report, cls_confusion = _evaluate_classification_records(
        test["records"], classes, prior_weights, best_cls_params
    )
    refine_seconds = time.perf_counter() - start

    run_dir = dirs["analysis"] / "ours" / "refined" / f"{now_stamp()}_{REFINED_METHOD}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")
    for common_name in ["history.csv", "training_curves.png", "best.pt"]:
        source_file = source / common_name
        if source_file.exists():
            shutil.copy2(source_file, run_dir / common_name)

    pd.DataFrame(detection_sweep_rows).sort_values(["val_map50", "val_map50_95"], ascending=False).to_csv(
        run_dir / "detection_refinement_sweep_val.csv", index=False
    )
    pd.DataFrame(classification_sweep_rows).sort_values(["val_macro_f1", "val_balanced_accuracy"], ascending=False).to_csv(
        run_dir / "classification_refinement_sweep_val.csv", index=False
    )
    test["segmentation_per_class"].to_csv(run_dir / "segmentation_per_class_test.csv", index=False)
    test["segmentation_per_image"].to_csv(run_dir / "segmentation_per_image_test.csv", index=False)
    test["counting_predictions"].to_csv(run_dir / "counting_predictions_test.csv", index=False)
    det_per_class.to_csv(run_dir / "detection_per_class_test.csv", index=False)
    det_named = det_per_class.copy()
    det_named["class_name"] = det_named["class_id"].map({idx + 1: name for idx, name in enumerate(classes)})
    det_named.to_csv(run_dir / "detection_per_class_test_named.csv", index=False)
    det_predictions.to_csv(run_dir / "detection_predictions_test.csv", index=False)
    cls_predictions.to_csv(run_dir / "classification_predictions_test.csv", index=False)
    cls_report.to_csv(run_dir / "classification_report_test.csv", index=False)
    cls_confusion.to_csv(run_dir / "confusion_matrix_test.csv")
    save_confusion_matrix(cls_confusion, run_dir / "confusion_matrix_test.png", title=f"{REFINED_DISPLAY}: ROI Classification")
    if test["records"]:
        first = test["records"][0]
        with Image.open(first["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, first["true_mask"], first["pred_mask"], run_dir / "sample_prediction_overlay.jpg")

    source_common = _load_source_common(source)
    common = {**source_common, "refine_seconds": refine_seconds}
    combined = {
        "classification": {**classification, **common},
        "counting": {**test["counting"], **common},
        "segmentation": {**test["segmentation"], **common},
        "detection": {**detection, **common},
    }
    json_dump(combined, run_dir / "test_metrics_by_task.json")
    json_dump(
        {
            "method": REFINED_METHOD,
            "display_name": REFINED_DISPLAY,
            "source_run": str(source.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "selected_detection_params": best_det_params,
            "selected_detection_val_metrics": best_det_metrics,
            "selected_classification_params": best_cls_params,
            "selected_classification_val_metrics": best_cls_metrics,
            "median_instance_area_512": median_area,
            "classification_prior_weights": prior_weights.tolist(),
            "refine_seconds": refine_seconds,
        },
        run_dir / "metadata.json",
    )

    base_meta = _metadata("multitask", seed, device, image_size, model_name, parameters)
    base_meta.update(
        {
            "method": REFINED_METHOD,
            "display_name": REFINED_DISPLAY,
            "family": "Unified multitask dense prediction + instance refinement",
            "refined_from": str(source.resolve()),
            "inference_refinement": True,
        }
    )
    _mirror_task_outputs(
        config,
        run_dir,
        "classification",
        {**classification, **common},
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
    _mirror_task_outputs(
        config,
        run_dir,
        "detection",
        {**detection, **common},
        {**base_meta, "task": "detection"},
        {
            "predictions_test.csv": run_dir / "detection_predictions_test.csv",
            "per_class_test.csv": run_dir / "detection_per_class_test_named.csv",
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir
