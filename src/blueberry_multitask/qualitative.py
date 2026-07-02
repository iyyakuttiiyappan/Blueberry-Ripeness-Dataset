from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from .annotations import PALETTE, prepare_annotations
from .config import output_dirs
from .datasets import IMAGENET_MEAN, IMAGENET_STD
from .metrics import box_iou
from .models import create_segmentation_model
from .utils import resolve_device


PRIMARY_METRICS = {
    "classification": ("macro_f1", False),
    "counting": ("mae", True),
    "segmentation": ("miou_foreground", False),
    "detection": ("map50", False),
}

CLASS_COLORS = {
    "green_immature": (64, 160, 43),
    "pale_pink": (241, 146, 178),
    "pink_turns_purple": (138, 79, 184),
    "fully_ripe": (46, 96, 200),
    "over_ripe": (185, 57, 63),
}


def _task_cfg(config: dict[str, Any], task: str, method: str) -> dict[str, Any]:
    cfg = dict(config.get("task_defaults", {}).get(task, {}))
    cfg.update(config.get("tasks", {}).get(task, {}).get(method, {}))
    return cfg


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _ensure_dirs(dirs: dict[str, Path]) -> dict[str, Path]:
    analysis = dirs["analysis"]
    paths = {
        "analysis": analysis,
        "tables": analysis / "tables",
        "figures": analysis / "figures",
        "dataset_figures": analysis / "figures" / "dataset_statistics",
        "qualitative_figures": analysis / "figures" / "qualitative",
        "failure_figures": analysis / "figures" / "failure_analysis",
        "paper_dataset": dirs["paper_ready"] / "dataset_statistics",
        "paper_qualitative": dirs["paper_ready"] / "qualitative",
        "paper_failure": dirs["paper_ready"] / "failure_analysis",
        "paper_tables": dirs["paper_ready"] / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _summary_table(dirs: dict[str, Path]) -> pd.DataFrame:
    summary_path = dirs["tables"] / "all_task_runs.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")
    return pd.read_csv(summary_path)


def _best_run(summary: pd.DataFrame, task: str) -> pd.Series:
    metric, lower = PRIMARY_METRICS[task]
    frame = summary[summary["task"] == task].copy()
    if frame.empty:
        raise RuntimeError(f"No completed runs found for {task}.")
    frame = frame.sort_values(metric, ascending=lower)
    return frame.iloc[0]


def _run_image_size(config: dict[str, Any], task: str, run: Any) -> int:
    value = getattr(run, "image_size", None)
    if value is not None and not pd.isna(value):
        return int(value)
    method_cfg = _task_cfg(config, task, str(run.method))
    return int(method_cfg.get("image_size", config.get("task_defaults", {}).get(task, {}).get("image_size", 512)))


def _copy_outputs(paths: dict[str, Path]) -> None:
    copy_pairs = [
        (paths["dataset_figures"], paths["paper_dataset"]),
        (paths["qualitative_figures"], paths["paper_qualitative"]),
        (paths["failure_figures"], paths["paper_failure"]),
    ]
    for source_dir, target_dir in copy_pairs:
        for source in source_dir.glob("*"):
            if source.is_file():
                shutil.copy2(source, target_dir / source.name)
    for source in paths["tables"].glob("*"):
        if source.is_file():
            shutil.copy2(source, paths["paper_tables"] / source.name)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _caption_panel(image: Image.Image, title: str, width: int, caption_height: int = 78) -> Image.Image:
    image = image.convert("RGB")
    scale = width / image.width
    resized = image.resize((width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, resized.height + caption_height), "white")
    canvas.paste(resized, (0, caption_height))
    draw = ImageDraw.Draw(canvas)
    font = _font(18, bold=True)
    small = _font(14)
    lines = _wrap_text(draw, title, font, width - 18)
    y = 8
    for idx, line in enumerate(lines):
        draw.text((9, y), line, fill=(25, 25, 25), font=font if idx == 0 else small)
        y += 20
    return canvas


def _grid(images: list[Image.Image], titles: list[str], path: Path, cols: int = 2, cell_width: int = 760) -> None:
    if not images:
        return
    panels = [_caption_panel(image, title, cell_width) for image, title in zip(images, titles)]
    rows = math.ceil(len(panels) / cols)
    row_heights = []
    for row_idx in range(rows):
        row_panels = panels[row_idx * cols : (row_idx + 1) * cols]
        row_heights.append(max(panel.height for panel in row_panels))
    gutter = 18
    canvas = Image.new(
        "RGB",
        (cols * cell_width + (cols - 1) * gutter, sum(row_heights) + (rows - 1) * gutter),
        "white",
    )
    y = 0
    for row_idx in range(rows):
        x = 0
        for panel in panels[row_idx * cols : (row_idx + 1) * cols]:
            canvas.paste(panel, (x, y))
            x += cell_width + gutter
        y += row_heights[row_idx] + gutter
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def _semantic_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.42) -> Image.Image:
    resized = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized.convert("RGB")).astype(np.float32)
    color = np.zeros_like(rgb)
    for label, palette_color in PALETTE.items():
        if label == 0:
            continue
        color[mask == label] = palette_color
    active = mask > 0
    blended = rgb.copy()
    blended[active] = (1.0 - alpha) * rgb[active] + alpha * color[active]
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _colorize_mask(mask: np.ndarray) -> Image.Image:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        canvas[mask == label] = color
    return Image.fromarray(canvas)


def _error_overlay(image: Image.Image, truth: np.ndarray, pred: np.ndarray) -> Image.Image:
    resized = image.resize((truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized.convert("RGB")).astype(np.float32)
    error = rgb.copy()
    truth_fg = truth > 0
    pred_fg = pred > 0
    tp = truth_fg & pred_fg & (truth == pred)
    class_confusion = truth_fg & pred_fg & (truth != pred)
    fn = truth_fg & ~pred_fg
    fp = ~truth_fg & pred_fg
    error[tp] = (0.55 * error[tp] + 0.45 * np.array([73, 170, 90])).clip(0, 255)
    error[class_confusion] = (0.45 * error[class_confusion] + 0.55 * np.array([245, 180, 70])).clip(0, 255)
    error[fn] = (0.35 * error[fn] + 0.65 * np.array([40, 100, 220])).clip(0, 255)
    error[fp] = (0.35 * error[fp] + 0.65 * np.array([220, 55, 60])).clip(0, 255)
    return Image.fromarray(error.clip(0, 255).astype(np.uint8))


def _concat_labeled(parts: list[tuple[str, Image.Image]], width_each: int = 300) -> Image.Image:
    label_h = 34
    font = _font(17, bold=True)
    resized: list[Image.Image] = []
    for _, image in parts:
        scale = width_each / image.width
        resized.append(image.resize((width_each, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS))
    height = max(image.height for image in resized) + label_h
    canvas = Image.new("RGB", (width_each * len(parts), height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, ((label, _), image) in enumerate(zip(parts, resized)):
        x = idx * width_each
        canvas.paste(image, (x, label_h))
        draw.text((x + 8, 8), label, fill=(20, 20, 20), font=font)
    return canvas


def dataset_statistics(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    dirs = output_dirs(config)
    classes = list(config["classes"])
    image_df = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")

    split_counts = image_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).reset_index()
    split_counts.columns = ["split", "images"]
    split_counts.to_csv(paths["tables"] / "dataset_split_counts.csv", index=False)

    class_split_rows = []
    for split, frame in image_df.groupby("split"):
        for class_name in classes:
            class_split_rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "workbook_count": int(frame[class_name].sum()),
                    "component_count": int(frame[f"{class_name}_components"].sum()),
                    "pixel_count": int(frame[f"{class_name}_pixels"].sum()),
                }
            )
    class_split = pd.DataFrame(class_split_rows)
    class_split.to_csv(paths["tables"] / "dataset_class_distribution_by_split.csv", index=False)

    area_stats = (
        instances.assign(
            bbox_width=instances["x2"] - instances["x1"],
            bbox_height=instances["y2"] - instances["y1"],
            bbox_area=lambda df: df["bbox_width"] * df["bbox_height"],
        )
        .groupby("class_name")
        .agg(
            instances=("area", "size"),
            area_mean=("area", "mean"),
            area_median=("area", "median"),
            area_p10=("area", lambda values: np.percentile(values, 10)),
            area_p90=("area", lambda values: np.percentile(values, 90)),
            bbox_width_median=("bbox_width", "median"),
            bbox_height_median=("bbox_height", "median"),
        )
        .reset_index()
    )
    area_stats.to_csv(paths["tables"] / "dataset_instance_area_stats.csv", index=False)

    image_stats = image_df[
        [
            "filename",
            "split",
            "Total",
            "class_component_total",
            "overall_component_total",
            "foreground_pixels",
            "class_pixels_total",
            "mask_conflict_pixels",
            "aligned_width",
            "aligned_height",
        ]
    ].copy()
    image_stats["count_component_abs_diff"] = (image_stats["Total"] - image_stats["class_component_total"]).abs()
    image_stats["foreground_fraction"] = image_stats["foreground_pixels"] / (
        image_stats["aligned_width"] * image_stats["aligned_height"]
    )
    image_stats.to_csv(paths["tables"] / "dataset_image_level_statistics.csv", index=False)

    fig_path = paths["dataset_figures"] / "split_image_counts.png"
    plt.figure(figsize=(7.5, 5))
    plt.bar(split_counts["split"], split_counts["images"], color=["#4c78a8", "#59a14f", "#f58518"])
    plt.title("Images by Split")
    plt.ylabel("Images")
    for idx, value in enumerate(split_counts["images"]):
        plt.text(idx, value + 2, str(int(value)), ha="center", fontsize=10)
    _save_fig(fig_path)

    pivot = class_split.pivot(index="split", columns="class_name", values="workbook_count").reindex(["train", "val", "test"])
    fig_path = paths["dataset_figures"] / "class_distribution_by_split.png"
    pivot.plot(kind="bar", stacked=True, figsize=(10, 5.5), color=[tuple(v / 255 for v in CLASS_COLORS[c]) for c in classes])
    plt.title("Workbook Berry Counts by Split and Class")
    plt.ylabel("Berries")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_fig(fig_path)

    fig_path = paths["dataset_figures"] / "count_distribution_by_split.png"
    plt.figure(figsize=(9, 5.5))
    for split, color in zip(["train", "val", "test"], ["#4c78a8", "#59a14f", "#f58518"]):
        values = image_df.loc[image_df["split"] == split, "Total"]
        plt.hist(values, bins=22, alpha=0.5, label=split, color=color)
    plt.title("Berry Count Distribution by Split")
    plt.xlabel("Workbook total berries per image")
    plt.ylabel("Images")
    plt.legend()
    _save_fig(fig_path)

    fig_path = paths["dataset_figures"] / "instance_area_distribution.png"
    data = [np.log10(instances.loc[instances["class_name"] == class_name, "area"].clip(lower=1)) for class_name in classes]
    plt.figure(figsize=(10, 5.5))
    plt.boxplot(data, labels=classes, showfliers=False)
    plt.title("Instance Area Distribution by Class")
    plt.ylabel("log10(mask area in pixels)")
    plt.xticks(rotation=25, ha="right")
    _save_fig(fig_path)

    sample = instances.sample(n=min(4000, len(instances)), random_state=13).copy()
    sample["bbox_width"] = sample["x2"] - sample["x1"]
    sample["bbox_height"] = sample["y2"] - sample["y1"]
    fig_path = paths["dataset_figures"] / "bbox_size_scatter.png"
    plt.figure(figsize=(7.5, 6.5))
    for class_name in classes:
        frame = sample[sample["class_name"] == class_name]
        color = tuple(v / 255 for v in CLASS_COLORS[class_name])
        plt.scatter(frame["bbox_width"], frame["bbox_height"], s=9, alpha=0.35, label=class_name, color=color)
    plt.title("Bounding Box Size Distribution")
    plt.xlabel("Box width (pixels)")
    plt.ylabel("Box height (pixels)")
    plt.legend(markerscale=2, fontsize=8)
    _save_fig(fig_path)

    presence = (image_df[classes] > 0).astype(int)
    cooccurrence = presence.T @ presence
    fig_path = paths["dataset_figures"] / "class_cooccurrence_heatmap.png"
    plt.figure(figsize=(8, 6.5))
    plt.imshow(cooccurrence, cmap="YlGnBu")
    plt.title("Class Co-occurrence Across Images")
    plt.xticks(range(len(classes)), classes, rotation=30, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(int(cooccurrence.iloc[i, j])), ha="center", va="center", fontsize=9)
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_fig(fig_path)

    fig_path = paths["dataset_figures"] / "mask_pixel_fraction_by_class.png"
    pixel_totals = image_df[[f"{class_name}_pixels" for class_name in classes]].sum()
    pixel_totals.index = classes
    total_pixels = float((image_df["aligned_width"] * image_df["aligned_height"]).sum())
    fractions = pixel_totals / total_pixels * 100.0
    plt.figure(figsize=(9, 5))
    plt.bar(fractions.index, fractions.values, color=[tuple(v / 255 for v in CLASS_COLORS[c]) for c in classes])
    plt.title("Annotated Pixel Fraction by Class")
    plt.ylabel("Percent of all image pixels")
    plt.xticks(rotation=25, ha="right")
    _save_fig(fig_path)

    fig_path = paths["dataset_figures"] / "count_component_agreement.png"
    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(image_df["Total"], image_df["class_component_total"], s=22, alpha=0.65, color="#4c78a8")
    limit = max(image_df["Total"].max(), image_df["class_component_total"].max()) + 5
    plt.plot([0, limit], [0, limit], color="black", linewidth=1)
    plt.title("Workbook Count vs Mask Components")
    plt.xlabel("Workbook total")
    plt.ylabel("Class-mask connected components")
    _save_fig(fig_path)

    examples = pd.concat(
        [
            image_df.sort_values("Total").head(1),
            image_df.iloc[(image_df["Total"] - image_df["Total"].median()).abs().sort_values().head(1).index],
            image_df.sort_values("Total", ascending=False).head(2),
        ]
    ).drop_duplicates("stem")
    panels: list[Image.Image] = []
    titles: list[str] = []
    for row in examples.itertuples(index=False):
        image = _read_rgb(row.image_path)
        mask = _read_mask(row.semantic_mask_path)
        overlay = _semantic_overlay(image, mask)
        panels.append(_concat_labeled([("RGB", image), ("Mask overlay", overlay)], width_each=300))
        titles.append(f"{row.filename}: total {row.Total}, split {row.split}")
    _grid(panels, titles, paths["dataset_figures"] / "dataset_example_overlays.jpg", cols=2, cell_width=650)

    return {
        "images": int(len(image_df)),
        "instances": int(len(instances)),
        "mean_count": float(image_df["Total"].mean()),
        "median_count": float(image_df["Total"].median()),
        "max_count": int(image_df["Total"].max()),
    }


def classification_analysis(
    config: dict[str, Any],
    summary: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    classes = list(config["classes"])
    best = _best_run(summary, "classification")
    best_run_dir = Path(str(best["run_dir"]))
    best_preds = pd.read_csv(best_run_dir / "predictions_test.csv")
    best_preds["abs_wrong"] = best_preds["y_true"] != best_preds["y_pred"]
    best_preds["true_class"] = best_preds["y_true"].map(dict(enumerate(classes)))
    best_preds["pred_class"] = best_preds["y_pred"].map(dict(enumerate(classes)))
    best_preds.to_csv(paths["tables"] / "classification_best_model_predictions.csv", index=False)

    correct_images: list[Image.Image] = []
    correct_titles: list[str] = []
    for class_name in classes:
        rows = best_preds[(best_preds["true_class"] == class_name) & (~best_preds["abs_wrong"])].sort_values(
            "confidence", ascending=False
        )
        for row in rows.head(2).itertuples(index=False):
            correct_images.append(_read_rgb(row.path))
            correct_titles.append(f"{class_name}, confidence {row.confidence:.2f}")
    _grid(
        correct_images[:10],
        correct_titles[:10],
        paths["qualitative_figures"] / "classification_correct_gallery.jpg",
        cols=5,
        cell_width=210,
    )

    wrong = best_preds[best_preds["abs_wrong"]].sort_values("confidence", ascending=False)
    failure_images: list[Image.Image] = []
    failure_titles: list[str] = []
    selected = []
    for class_name in classes:
        selected.append(wrong[wrong["true_class"] == class_name].head(2))
    wrong_selected = pd.concat(selected).drop_duplicates("path").head(12) if selected else wrong.head(12)
    for row in wrong_selected.itertuples(index=False):
        failure_images.append(_read_rgb(row.path))
        failure_titles.append(f"{row.true_class} -> {row.pred_class}, confidence {row.confidence:.2f}")
    _grid(
        failure_images,
        failure_titles,
        paths["qualitative_figures"] / "classification_failure_gallery.jpg",
        cols=4,
        cell_width=240,
    )

    aggregate_rows = []
    pair_rows = []
    hard_rows = []
    for run in summary[summary["task"] == "classification"].itertuples(index=False):
        run_dir = Path(str(run.run_dir))
        pred_path = run_dir / "predictions_test.csv"
        if not pred_path.exists():
            continue
        preds = pd.read_csv(pred_path)
        preds["true_class"] = preds["y_true"].map(dict(enumerate(classes)))
        preds["pred_class"] = preds["y_pred"].map(dict(enumerate(classes)))
        preds["wrong"] = preds["y_true"] != preds["y_pred"]
        for class_name, frame in preds.groupby("true_class"):
            aggregate_rows.append(
                {
                    "method": run.method,
                    "display_name": run.display_name,
                    "true_class": class_name,
                    "support": int(len(frame)),
                    "errors": int(frame["wrong"].sum()),
                    "error_rate": float(frame["wrong"].mean()) if len(frame) else 0.0,
                }
            )
        pairs = preds[preds["wrong"]].groupby(["true_class", "pred_class"]).size().reset_index(name="count")
        for row in pairs.itertuples(index=False):
            pair_rows.append(
                {
                    "method": run.method,
                    "display_name": run.display_name,
                    "true_class": row.true_class,
                    "pred_class": row.pred_class,
                    "count": int(row.count),
                }
            )
        top_wrong = preds[preds["wrong"]].sort_values("confidence", ascending=False).head(25)
        for row in top_wrong.itertuples(index=False):
            hard_rows.append(
                {
                    "method": run.method,
                    "display_name": run.display_name,
                    "path": row.path,
                    "true_class": row.true_class,
                    "pred_class": row.pred_class,
                    "confidence": float(row.confidence),
                }
            )

    aggregate = pd.DataFrame(aggregate_rows)
    pairs = pd.DataFrame(pair_rows)
    hard = pd.DataFrame(hard_rows)
    aggregate.to_csv(paths["tables"] / "classification_error_by_class.csv", index=False)
    pairs.to_csv(paths["tables"] / "classification_confusion_pairs.csv", index=False)
    hard.to_csv(paths["tables"] / "classification_high_confidence_failures.csv", index=False)

    heat = aggregate.pivot_table(index="display_name", columns="true_class", values="error_rate", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(10, max(4.5, len(heat) * 0.42)))
    plt.imshow(heat, cmap="Reds", vmin=0, vmax=max(0.01, float(heat.to_numpy().max())))
    plt.title("Classification Error Rate by True Class")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            plt.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_fig(paths["failure_figures"] / "classification_error_rate_heatmap.png")

    matrix = pd.crosstab(best_preds["true_class"], best_preds["pred_class"]).reindex(index=classes, columns=classes, fill_value=0)
    norm = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    plt.figure(figsize=(8, 6.8))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Normalized Confusion Matrix: {best['display_name']}")
    plt.xticks(range(len(classes)), classes, rotation=25, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{norm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_fig(paths["failure_figures"] / "classification_best_confusion_normalized.png")

    return {
        "best_method": str(best["display_name"]),
        "best_macro_f1": float(best["macro_f1"]),
        "wrong_examples": int(best_preds["abs_wrong"].sum()),
    }


def counting_analysis(summary: pd.DataFrame, paths: dict[str, Path]) -> dict[str, Any]:
    best = _best_run(summary, "counting")
    best_run_dir = Path(str(best["run_dir"]))
    best_preds = pd.read_csv(best_run_dir / "predictions_test.csv")
    best_preds["abs_error"] = best_preds["error"].abs()
    best_preds.to_csv(paths["tables"] / "counting_best_model_predictions.csv", index=False)

    easy = best_preds.assign(bin=pd.qcut(best_preds["y_true"], q=4, duplicates="drop")).sort_values("abs_error")
    easy = easy.groupby("bin", observed=False).head(2).head(8)
    easy_images = [_read_rgb(row.path) for row in easy.itertuples(index=False)]
    easy_titles = [f"true {row.y_true:.0f}, pred {row.y_pred:.1f}, error {row.error:+.1f}" for row in easy.itertuples(index=False)]
    _grid(easy_images, easy_titles, paths["qualitative_figures"] / "counting_representative_examples.jpg", cols=4, cell_width=330)

    failures = best_preds.sort_values("abs_error", ascending=False).head(8)
    failure_images = [_read_rgb(row.path) for row in failures.itertuples(index=False)]
    failure_titles = [f"true {row.y_true:.0f}, pred {row.y_pred:.1f}, error {row.error:+.1f}" for row in failures.itertuples(index=False)]
    _grid(failure_images, failure_titles, paths["qualitative_figures"] / "counting_failure_examples.jpg", cols=4, cell_width=330)

    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(best_preds["y_true"], best_preds["y_pred"], s=28, alpha=0.72, color="#4c78a8")
    limit = max(best_preds["y_true"].max(), best_preds["y_pred"].max()) + 5
    plt.plot([0, limit], [0, limit], color="black", linewidth=1)
    plt.title(f"Predicted vs True Counts: {best['display_name']}")
    plt.xlabel("True count")
    plt.ylabel("Predicted count")
    _save_fig(paths["failure_figures"] / "counting_predicted_vs_true_best.png")

    plt.figure(figsize=(7.5, 5.5))
    plt.axhline(0, color="black", linewidth=1)
    plt.scatter(best_preds["y_true"], best_preds["error"], s=28, alpha=0.72, color="#f58518")
    plt.title(f"Counting Residuals: {best['display_name']}")
    plt.xlabel("True count")
    plt.ylabel("Prediction error")
    _save_fig(paths["failure_figures"] / "counting_residuals_best.png")

    per_image_rows = []
    for run in summary[summary["task"] == "counting"].itertuples(index=False):
        run_dir = Path(str(run.run_dir))
        pred_path = run_dir / "predictions_test.csv"
        if not pred_path.exists():
            continue
        preds = pd.read_csv(pred_path)
        preds["abs_error"] = preds["error"].abs()
        preds["display_name"] = run.display_name
        preds["method"] = run.method
        per_image_rows.append(preds)
    all_preds = pd.concat(per_image_rows, ignore_index=True)
    all_preds.to_csv(paths["tables"] / "counting_per_image_failures.csv", index=False)
    bins = pd.qcut(all_preds["y_true"], q=4, duplicates="drop")
    all_preds["count_bin"] = bins.astype(str)
    bin_mae = all_preds.groupby(["display_name", "count_bin"], observed=False)["abs_error"].mean().reset_index()
    bin_mae.to_csv(paths["tables"] / "counting_mae_by_count_bin.csv", index=False)
    heat = bin_mae.pivot_table(index="display_name", columns="count_bin", values="abs_error", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(10, max(4.5, len(heat) * 0.42)))
    plt.imshow(heat, cmap="OrRd")
    plt.title("Counting MAE by True-count Bin")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            plt.text(j, i, f"{heat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_fig(paths["failure_figures"] / "counting_mae_by_count_bin_heatmap.png")

    return {
        "best_method": str(best["display_name"]),
        "best_mae": float(best["mae"]),
        "worst_abs_error": float(best_preds["abs_error"].max()),
    }


def _scaled_gt_for_detection(
    test_images: pd.DataFrame,
    instances: pd.DataFrame,
    image_size: int,
) -> dict[int, dict[str, np.ndarray]]:
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=False)}
    output: dict[int, dict[str, np.ndarray]] = {}
    for idx, row in enumerate(test_images.itertuples(index=False)):
        group = grouped.get(row.stem, pd.DataFrame())
        if group.empty:
            output[idx] = {"boxes": np.zeros((0, 4), dtype=float), "labels": np.zeros((0,), dtype=int)}
            continue
        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).copy()
        boxes[:, [0, 2]] *= image_size / float(row.aligned_width)
        boxes[:, [1, 3]] *= image_size / float(row.aligned_height)
        output[idx] = {"boxes": boxes, "labels": group["det_label"].to_numpy(dtype=int)}
    return output


def _match_detection(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.50,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    order = np.argsort(-pred_scores) if len(pred_scores) else np.asarray([], dtype=int)
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    pred_rows: list[dict[str, Any]] = []
    tp = fp = 0
    for pred_idx in order:
        same_class = np.where((gt_labels == pred_labels[pred_idx]) & (~matched_gt))[0]
        best_gt = -1
        best_iou = 0.0
        if len(same_class):
            overlaps = box_iou(pred_boxes[pred_idx : pred_idx + 1], gt_boxes[same_class])[0]
            local_best = int(np.argmax(overlaps))
            best_iou = float(overlaps[local_best])
            best_gt = int(same_class[local_best])
        is_tp = best_gt >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched_gt[best_gt] = True
            tp += 1
        else:
            fp += 1
        pred_rows.append(
            {
                "pred_index": int(pred_idx),
                "pred_label": int(pred_labels[pred_idx]),
                "score": float(pred_scores[pred_idx]),
                "best_iou": best_iou,
                "matched_gt_index": best_gt,
                "match_type": "TP" if is_tp else "FP",
            }
        )
    fn = int((~matched_gt).sum())
    stats = {"tp": int(tp), "fp": int(fp), "fn": fn, "gt": int(len(gt_boxes)), "pred": int(len(pred_boxes))}
    return stats, pred_rows, matched_gt


def _draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    scores: np.ndarray | None = None,
    width: int = 4,
    show_labels: bool = True,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(15, bold=True)
    for idx, box in enumerate(boxes):
        label_id = int(labels[idx])
        class_name = classes[label_id - 1] if 1 <= label_id <= len(classes) else str(label_id)
        color = CLASS_COLORS.get(class_name, (240, 120, 40))
        x1, y1, x2, y2 = [float(v) for v in box]
        for offset in range(width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        if not show_labels:
            continue
        text = class_name
        if scores is not None:
            text = f"{class_name} {float(scores[idx]):.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tx, ty = int(x1), max(0, int(y1) - (bbox[3] - bbox[1]) - 4)
        draw.rectangle((tx, ty, tx + bbox[2] - bbox[0] + 6, ty + bbox[3] - bbox[1] + 4), fill=color)
        draw.text((tx + 3, ty + 2), text, fill="white", font=font)
    return canvas


def detection_analysis(
    config: dict[str, Any],
    summary: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    dirs = output_dirs(config)
    classes = list(config["classes"])
    image_df = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    test_images = image_df[image_df["split"] == "test"].reset_index(drop=True)

    best = _best_run(summary, "detection")
    best_size = int(best["image_size"]) if "image_size" in best and not pd.isna(best["image_size"]) else int(
        _task_cfg(config, "detection", str(best["method"])).get("image_size", 768)
    )
    best_gt = _scaled_gt_for_detection(test_images, instances[instances["split"] == "test"], best_size)

    per_image_rows = []
    per_class_rows = []
    predictions_by_best_index: dict[int, dict[str, np.ndarray]] = {}
    for run in summary[summary["task"] == "detection"].itertuples(index=False):
        run_dir = Path(str(run.run_dir))
        pred_path = run_dir / "predictions_test.csv"
        if not pred_path.exists():
            continue
        image_size = _run_image_size(config, "detection", run)
        gt = _scaled_gt_for_detection(test_images, instances[instances["split"] == "test"], image_size)
        pred_df = pd.read_csv(pred_path)
        class_totals = {
            class_id: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
            for class_id in range(1, len(classes) + 1)
        }
        for image_idx, image_row in enumerate(test_images.itertuples(index=False)):
            frame = pred_df[pred_df["batch_image_index"] == image_idx]
            pred_boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if not frame.empty else np.zeros((0, 4), dtype=float)
            pred_labels = frame["label"].to_numpy(dtype=int) if not frame.empty else np.zeros((0,), dtype=int)
            pred_scores = frame["score"].to_numpy(dtype=float) if not frame.empty else np.zeros((0,), dtype=float)
            target = gt[image_idx]
            stats, pred_matches, matched_gt = _match_detection(
                pred_boxes,
                pred_labels,
                pred_scores,
                target["boxes"],
                target["labels"],
            )
            precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
            recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
            per_image_rows.append(
                {
                    "method": run.method,
                    "display_name": run.display_name,
                    "image_index": image_idx,
                    "filename": image_row.filename,
                    "image_path": image_row.image_path,
                    **stats,
                    "precision50": precision,
                    "recall50": recall,
                    "failure_score": stats["fp"] + stats["fn"],
                }
            )
            for class_id in range(1, len(classes) + 1):
                class_totals[class_id]["gt"] += int((target["labels"] == class_id).sum())
                class_totals[class_id]["pred"] += int((pred_labels == class_id).sum())
                class_totals[class_id]["fn"] += int(((target["labels"] == class_id) & (~matched_gt)).sum())
            for match in pred_matches:
                class_id = int(match["pred_label"])
                if class_id in class_totals:
                    class_totals[class_id]["tp" if match["match_type"] == "TP" else "fp"] += 1
            if str(run.method) == str(best["method"]):
                predictions_by_best_index[image_idx] = {
                    "boxes": pred_boxes,
                    "labels": pred_labels,
                    "scores": pred_scores,
                }
        for class_id, values in class_totals.items():
            per_class_rows.append(
                {
                    "method": run.method,
                    "display_name": run.display_name,
                    "class_id": class_id,
                    "class_name": classes[class_id - 1],
                    **values,
                    "fn_rate": values["fn"] / max(1, values["gt"]),
                    "fp_per_gt": values["fp"] / max(1, values["gt"]),
                }
            )

    per_image = pd.DataFrame(per_image_rows)
    per_class = pd.DataFrame(per_class_rows)
    per_image.to_csv(paths["tables"] / "detection_per_image_failures.csv", index=False)
    per_class.to_csv(paths["tables"] / "detection_failure_by_class.csv", index=False)

    fn_heat = per_class.pivot_table(index="display_name", columns="class_name", values="fn_rate", aggfunc="mean").fillna(0.0)
    fp_heat = per_class.pivot_table(index="display_name", columns="class_name", values="fp_per_gt", aggfunc="mean").fillna(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(fn_heat) * 0.43)))
    for ax, heat, title, cmap in [
        (axes[0], fn_heat, "False-negative Rate by Class", "Blues"),
        (axes[1], fp_heat, "False Positives per GT by Class", "Reds"),
    ]:
        im = ax.imshow(heat, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        for i in range(len(heat.index)):
            for j in range(len(heat.columns)):
                ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(paths["failure_figures"] / "detection_fp_fn_by_class_heatmaps.png", dpi=180)
    plt.close(fig)

    best_per_image = per_image[per_image["method"] == best["method"]].copy()
    representative = pd.concat(
        [
            best_per_image.sort_values("failure_score").head(2),
            best_per_image.sort_values("gt").tail(2),
        ]
    ).drop_duplicates("image_index").head(4)
    failure = best_per_image.sort_values(["failure_score", "fn", "fp"], ascending=False).head(4)

    def make_detection_panels(frame: pd.DataFrame, output_name: str) -> None:
        panels: list[Image.Image] = []
        titles: list[str] = []
        for row in frame.itertuples(index=False):
            target = best_gt[int(row.image_index)]
            pred = predictions_by_best_index.get(
                int(row.image_index),
                {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,))},
            )
            image = _read_rgb(row.image_path).resize((best_size, best_size), Image.Resampling.BILINEAR)
            gt_panel = _draw_boxes(image, target["boxes"], target["labels"], classes, width=3, show_labels=False)
            pred_panel = _draw_boxes(image, pred["boxes"], pred["labels"], classes, pred["scores"], width=3, show_labels=False)
            panels.append(_concat_labeled([("RGB", image), ("Ground truth", gt_panel), ("Prediction", pred_panel)], width_each=275))
            titles.append(
                f"{row.filename}: TP {row.tp}, FP {row.fp}, FN {row.fn}, precision {row.precision50:.2f}, recall {row.recall50:.2f}"
            )
        _grid(panels, titles, paths["qualitative_figures"] / output_name, cols=2, cell_width=880)

    make_detection_panels(representative, "detection_representative_examples.jpg")
    make_detection_panels(failure, "detection_failure_examples.jpg")

    return {
        "best_method": str(best["display_name"]),
        "best_map50": float(best["map50"]),
        "mean_failure_score": float(best_per_image["failure_score"].mean()),
    }


@torch.inference_mode()
def _segmentation_predict_best(
    config: dict[str, Any],
    best: pd.Series,
    test_images: pd.DataFrame,
    device_name: str | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    classes = list(config["classes"])
    method = str(best["method"])
    method_cfg = _task_cfg(config, "segmentation", method)
    image_size = int(method_cfg.get("image_size", 512))
    device = resolve_device(device_name)
    created = create_segmentation_model(method_cfg, num_classes=len(classes) + 1, pretrained=False)
    model = created.model.to(device)
    checkpoint = torch.load(Path(str(best["run_dir"])) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for row in tqdm(test_images.itertuples(index=False), total=len(test_images), desc="segmentation qualitative"):
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        truth = _read_mask(row.semantic_mask_path)
        truth = np.asarray(Image.fromarray(truth).resize((image_size, image_size), Image.Resampling.NEAREST))
        tensor = F.normalize(F.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
        output = model(tensor)
        logits = output["out"] if isinstance(output, dict) else output
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        predictions[str(row.stem)] = pred
        true_fg = truth > 0
        pred_fg = pred > 0
        intersection = int((true_fg & pred_fg).sum())
        union = int((true_fg | pred_fg).sum())
        foreground_iou = intersection / max(1, union)
        class_ious = {}
        for class_id, class_name in enumerate(classes, start=1):
            t = truth == class_id
            p = pred == class_id
            class_ious[f"iou_{class_name}"] = float((t & p).sum() / max(1, (t | p).sum()))
        rows.append(
            {
                "stem": row.stem,
                "filename": row.filename,
                "image_path": row.image_path,
                "semantic_mask_path": row.semantic_mask_path,
                "foreground_iou": foreground_iou,
                "true_foreground_pixels": int(true_fg.sum()),
                "pred_foreground_pixels": int(pred_fg.sum()),
                "foreground_pixel_error": int(pred_fg.sum() - true_fg.sum()),
                **class_ious,
            }
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), predictions


def segmentation_analysis(
    config: dict[str, Any],
    summary: pd.DataFrame,
    paths: dict[str, Path],
    device_name: str | None,
) -> dict[str, Any]:
    dirs = output_dirs(config)
    classes = list(config["classes"])
    image_df = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    test_images = image_df[image_df["split"] == "test"].reset_index(drop=True)
    best = _best_run(summary, "segmentation")

    per_class_rows = []
    for run in summary[summary["task"] == "segmentation"].itertuples(index=False):
        run_dir = Path(str(run.run_dir))
        per_class_path = run_dir / "per_class_test.csv"
        if not per_class_path.exists():
            continue
        frame = pd.read_csv(per_class_path)
        frame["method"] = run.method
        frame["display_name"] = run.display_name
        per_class_rows.append(frame)
    per_class = pd.concat(per_class_rows, ignore_index=True)
    per_class.to_csv(paths["tables"] / "segmentation_per_class_iou_all_methods.csv", index=False)
    heat = (
        per_class[per_class["class_name"] != "background"]
        .pivot_table(index="display_name", columns="class_name", values="iou", aggfunc="mean")
        .fillna(0.0)
    )
    plt.figure(figsize=(10, max(4.5, len(heat) * 0.42)))
    plt.imshow(heat, cmap="YlGnBu", vmin=0, vmax=max(0.01, float(heat.to_numpy().max())))
    plt.title("Segmentation IoU by Class")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            plt.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_fig(paths["failure_figures"] / "segmentation_per_class_iou_heatmap.png")

    per_image, predictions = _segmentation_predict_best(config, best, test_images, device_name)
    per_image.to_csv(paths["tables"] / "segmentation_best_model_per_image_iou.csv", index=False)
    method_cfg = _task_cfg(config, "segmentation", str(best["method"]))
    image_size = int(method_cfg.get("image_size", 512))

    representative = pd.concat(
        [
            per_image.sort_values("foreground_iou", ascending=False).head(2),
            per_image.iloc[(per_image["foreground_iou"] - per_image["foreground_iou"].median()).abs().sort_values().head(2).index],
        ]
    ).drop_duplicates("stem").head(4)
    failure = per_image.sort_values("foreground_iou").head(4)

    def make_segmentation_panels(frame: pd.DataFrame, output_name: str) -> None:
        panels: list[Image.Image] = []
        titles: list[str] = []
        for row in frame.itertuples(index=False):
            image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
            truth = _read_mask(row.semantic_mask_path)
            truth = np.asarray(Image.fromarray(truth).resize((image_size, image_size), Image.Resampling.NEAREST))
            pred = predictions[str(row.stem)]
            gt_overlay = _semantic_overlay(image, truth)
            pred_overlay = _semantic_overlay(image, pred)
            error = _error_overlay(image, truth, pred)
            panels.append(
                _concat_labeled(
                    [("RGB", image), ("Ground truth", gt_overlay), ("Prediction", pred_overlay), ("Error map", error)],
                    width_each=235,
                )
            )
            titles.append(f"{row.filename}: foreground IoU {row.foreground_iou:.3f}, pixel error {row.foreground_pixel_error:+d}")
        _grid(panels, titles, paths["qualitative_figures"] / output_name, cols=2, cell_width=980)

    make_segmentation_panels(representative, "segmentation_representative_examples.jpg")
    make_segmentation_panels(failure, "segmentation_failure_examples.jpg")

    return {
        "best_method": str(best["display_name"]),
        "best_miou_foreground": float(best["miou_foreground"]),
        "worst_foreground_iou": float(per_image["foreground_iou"].min()),
    }


def _write_report(
    paths: dict[str, Path],
    dataset: dict[str, Any],
    classification: dict[str, Any],
    counting: dict[str, Any],
    detection: dict[str, Any],
    segmentation: dict[str, Any],
) -> None:
    lines = [
        "# Fresh Benchmark Qualitative and Failure Analysis",
        "",
        "## Dataset Statistics",
        "",
        f"- Images: {dataset['images']}",
        f"- Instances/crops: {dataset['instances']}",
        f"- Mean berries per image: {dataset['mean_count']:.2f}",
        f"- Median berries per image: {dataset['median_count']:.0f}",
        f"- Maximum berries in one image: {dataset['max_count']}",
        "",
        "## Best-model Qualitative Summaries",
        "",
        f"- Classification: {classification['best_method']} macro F1 {classification['best_macro_f1']:.4f}; "
        f"{classification['wrong_examples']} test crop errors.",
        f"- Counting: {counting['best_method']} MAE {counting['best_mae']:.4f}; "
        f"worst absolute error {counting['worst_abs_error']:.2f}.",
        f"- Detection: {detection['best_method']} mAP50 {detection['best_map50']:.4f}; "
        f"mean image-level FP+FN {detection['mean_failure_score']:.2f}.",
        f"- Segmentation: {segmentation['best_method']} foreground mIoU {segmentation['best_miou_foreground']:.4f}; "
        f"worst image foreground IoU {segmentation['worst_foreground_iou']:.4f}.",
        "",
        "## Figure Folders",
        "",
        f"- Dataset statistics: `{paths['dataset_figures']}`",
        f"- Qualitative examples: `{paths['qualitative_figures']}`",
        f"- Failure analysis: `{paths['failure_figures']}`",
        "",
        "## Table Folder",
        "",
        f"- Tables: `{paths['tables']}`",
    ]
    report_path = paths["analysis"] / "qualitative_failure_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(report_path, paths["paper_tables"] / report_path.name)


def run_qualitative_analysis(config: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    prepare_annotations(config)
    dirs = output_dirs(config)
    paths = _ensure_dirs(dirs)
    summary = _summary_table(dirs)

    dataset = dataset_statistics(config, paths)
    classification = classification_analysis(config, summary, paths)
    counting = counting_analysis(summary, paths)
    detection = detection_analysis(config, summary, paths)
    segmentation = segmentation_analysis(config, summary, paths, device_name)
    _write_report(paths, dataset, classification, counting, detection, segmentation)
    _copy_outputs(paths)

    result = {
        "dataset": dataset,
        "classification": classification,
        "counting": counting,
        "detection": detection,
        "segmentation": segmentation,
        "analysis_root": str(paths["analysis"].resolve()),
        "paper_ready_qualitative": str(paths["paper_qualitative"].resolve()),
        "paper_ready_failure": str(paths["paper_failure"].resolve()),
        "paper_ready_dataset_statistics": str(paths["paper_dataset"].resolve()),
    }
    with (paths["analysis"] / "qualitative_failure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
