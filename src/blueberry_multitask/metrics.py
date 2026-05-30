from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def classification_report_df(y_true: list[int], y_pred: list[int], class_names: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index(names="class")


def confusion_df(y_true: list[int], y_pred: list[int], class_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def counting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true), 1.0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
        "bias": float(np.mean(y_pred - y_true)),
    }


class SegmentationMeter:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu().numpy().reshape(-1)
        true = target.detach().cpu().numpy().reshape(-1)
        valid = (true >= 0) & (true < self.num_classes)
        matrix = np.bincount(
            self.num_classes * true[valid].astype(np.int64) + pred[valid].astype(np.int64),
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += matrix

    def metrics(self, class_names: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
        tp = np.diag(self.confusion).astype(float)
        fp = self.confusion.sum(axis=0).astype(float) - tp
        fn = self.confusion.sum(axis=1).astype(float) - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)
        dice = np.divide(2 * tp, denom_dice, out=np.zeros_like(tp), where=denom_dice > 0)
        pixel_acc = float(tp.sum() / max(1.0, self.confusion.sum()))
        rows = []
        names = ["background", *class_names]
        for idx, name in enumerate(names):
            rows.append({"class_id": idx, "class_name": name, "iou": float(iou[idx]), "dice": float(dice[idx])})
        foreground = slice(1, None)
        metrics = {
            "pixel_accuracy": pixel_acc,
            "miou": float(np.mean(iou)),
            "miou_foreground": float(np.mean(iou[foreground])) if len(iou) > 1 else float(np.mean(iou)),
            "dice": float(np.mean(dice)),
            "dice_foreground": float(np.mean(dice[foreground])) if len(dice) > 1 else float(np.mean(dice)),
        }
        return metrics, pd.DataFrame(rows)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xa = np.maximum(x11[:, None], x21[None, :])
    ya = np.maximum(y11[:, None], y21[None, :])
    xb = np.minimum(x12[:, None], x22[None, :])
    yb = np.minimum(y12[:, None], y22[None, :])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = np.maximum(0, x12 - x11) * np.maximum(0, y12 - y11)
    area2 = np.maximum(0, x22 - x21) * np.maximum(0, y22 - y21)
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _ap_at_threshold(predictions: list[dict], targets: list[dict], class_id: int, iou_threshold: float) -> tuple[float, int, int, int]:
    gt_by_image: dict[int, np.ndarray] = {}
    matched_by_image: dict[int, np.ndarray] = {}
    for image_id, target in enumerate(targets):
        labels = target["labels"]
        boxes = target["boxes"]
        selected = labels == class_id
        gt_boxes = boxes[selected]
        gt_by_image[image_id] = gt_boxes
        matched_by_image[image_id] = np.zeros((len(gt_boxes),), dtype=bool)

    pred_rows = []
    for image_id, prediction in enumerate(predictions):
        labels = prediction["labels"]
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        selected = labels == class_id
        for box, score in zip(boxes[selected], scores[selected]):
            pred_rows.append((float(score), image_id, box))
    pred_rows.sort(key=lambda item: item[0], reverse=True)

    tp = np.zeros((len(pred_rows),), dtype=float)
    fp = np.zeros((len(pred_rows),), dtype=float)
    for idx, (_, image_id, box) in enumerate(pred_rows):
        gt_boxes = gt_by_image[image_id]
        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue
        overlaps = box_iou(np.asarray([box]), gt_boxes)[0]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not matched_by_image[image_id][best]:
            tp[idx] = 1.0
            matched_by_image[image_id][best] = True
        else:
            fp[idx] = 1.0

    total_gt = int(sum(len(value) for value in gt_by_image.values()))
    if total_gt == 0:
        return 0.0, 0, int(fp.sum()), 0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(1, total_gt)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)
    ap = _average_precision(recall, precision)
    return ap, int(tp.sum()), int(fp.sum()), total_gt


def detection_metrics(
    predictions: list[dict],
    targets: list[dict],
    num_classes: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    thresholds = np.arange(0.50, 1.00, 0.05)
    per_class_rows = []
    ap50_values = []
    ap5095_values = []
    total_tp50 = total_fp50 = total_gt = 0
    for class_id in range(1, num_classes + 1):
        aps = []
        tp50 = fp50 = gt50 = 0
        for threshold in thresholds:
            ap, tp, fp, gt = _ap_at_threshold(predictions, targets, class_id, float(threshold))
            aps.append(ap)
            if abs(threshold - 0.50) < 1e-9:
                tp50, fp50, gt50 = tp, fp, gt
        ap50 = aps[0]
        ap5095 = float(np.mean(aps))
        ap50_values.append(ap50)
        ap5095_values.append(ap5095)
        total_tp50 += tp50
        total_fp50 += fp50
        total_gt += gt50
        per_class_rows.append(
            {
                "class_id": class_id,
                "ap50": float(ap50),
                "ap50_95": ap5095,
                "tp50": int(tp50),
                "fp50": int(fp50),
                "gt": int(gt50),
            }
        )
    precision50 = total_tp50 / max(1, total_tp50 + total_fp50)
    recall50 = total_tp50 / max(1, total_gt)
    f1_50 = 2 * precision50 * recall50 / max(precision50 + recall50, 1e-9)
    metrics = {
        "map50": float(np.mean(ap50_values)),
        "map50_95": float(np.mean(ap5095_values)),
        "precision50": float(precision50),
        "recall50": float(recall50),
        "f1_50": float(f1_50),
    }
    return metrics, pd.DataFrame(per_class_rows)

