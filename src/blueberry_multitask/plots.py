from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .annotations import PALETTE


def save_history_plot(history: pd.DataFrame, path: str | Path, metric: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if "train_loss" in history:
        plt.plot(history["epoch"], history["train_loss"], label="train loss", color="#4c78a8")
    if "val_loss" in history:
        plt.plot(history["epoch"], history["val_loss"], label="val loss", color="#f58518")
    metric_col = f"val_{metric}"
    if metric_col in history:
        ax = plt.gca().twinx()
        ax.plot(history["epoch"], history[metric_col], label=metric_col, color="#54a24b")
        ax.set_ylabel(metric)
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_matrix(matrix: pd.DataFrame, path: str | Path, title: str = "Confusion Matrix") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    data = matrix.to_numpy()
    plt.imshow(data, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    max_value = data.max() if data.size else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > max_value / 2 else "black"
            plt.text(j, i, str(data[i, j]), ha="center", va="center", color=color, fontsize=9)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_prediction_overlay(image: Image.Image, target: np.ndarray, pred: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_arr = np.asarray(image.convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.Resampling.BILINEAR))

    def colorize(mask: np.ndarray) -> np.ndarray:
        canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in PALETTE.items():
            canvas[mask == label] = color
        return canvas

    target_color = colorize(target)
    pred_color = colorize(pred)
    target_blend = (0.55 * image_arr + 0.45 * target_color).clip(0, 255).astype(np.uint8)
    pred_blend = (0.55 * image_arr + 0.45 * pred_color).clip(0, 255).astype(np.uint8)
    panel = np.concatenate([image_arr, target_blend, pred_blend], axis=1)
    Image.fromarray(panel).save(path, quality=92)


def save_metric_bar(summary: pd.DataFrame, metric: str, path: str | Path, title: str) -> None:
    if summary.empty or metric not in summary.columns:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values(metric, ascending=True).tail(20)
    labels = frame.get("display_name", frame["method"]).astype(str)
    plt.figure(figsize=(10, max(5, len(frame) * 0.38)))
    plt.barh(labels, frame[metric], color="#4c78a8")
    plt.title(title)
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
