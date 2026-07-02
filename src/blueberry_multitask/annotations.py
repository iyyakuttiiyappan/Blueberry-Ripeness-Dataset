from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import output_dirs


PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (64, 160, 43),
    2: (241, 146, 178),
    3: (138, 79, 184),
    4: (46, 96, 200),
    5: (185, 57, 63),
}
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


@dataclass(frozen=True)
class PreparedPaths:
    image_manifest: Path
    instances: Path
    crops: Path
    audit_report: Path


def _normalise_exts(values: list[str]) -> set[str]:
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}


def _binary_mask_dir(data_root: Path, aliases: list[str]) -> Path:
    existing = {path.name.lower(): path for path in data_root.iterdir() if path.is_dir()}
    for alias in aliases:
        found = existing.get(alias.lower())
        if found is not None:
            return found
    raise FileNotFoundError(f"Could not find binary mask folder from aliases: {aliases}")


def _find_mask(mask_dir: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    candidates = [mask_dir / image_path.name]
    candidates.extend(mask_dir / f"{image_path.stem}{extension}" for extension in MASK_EXTENSIONS)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for {image_name} in {mask_dir}")


def _read_counts(workbook: Path, classes: list[str], image_stems: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_excel(workbook)
    if "Image Name" not in df.columns:
        raise ValueError(f"{workbook} must contain an 'Image Name' column.")
    missing_cols = [name for name in classes if name not in df.columns]
    if missing_cols:
        raise ValueError(f"{workbook} is missing class count columns: {missing_cols}")

    df = df.copy()
    df["filename"] = df["Image Name"].astype(str)
    df["stem"] = df["filename"].map(lambda value: Path(value).stem)
    df = df[df["stem"].isin(image_stems)].copy()
    for name in classes:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)
    if "Total" not in df.columns:
        df["Total"] = df[classes].sum(axis=1)
    else:
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(df[classes].sum(axis=1)).astype(int)
    df["computed_total"] = df[classes].sum(axis=1)
    audit = {
        "count_rows_kept": int(len(df)),
        "workbook_total_mismatch_rows": int((df["Total"] != df["computed_total"]).sum()),
    }
    return df, audit


def _fixed_split_from_manifest(image_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame | None:
    split_manifest = config.get("paths", {}).get("fixed_split_manifest")
    if not split_manifest:
        return None

    manifest_path = Path(split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Configured fixed split manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain a 'split' column.")
    if "stem" not in manifest.columns:
        if "image_id" in manifest.columns:
            manifest["stem"] = manifest["image_id"].astype(str)
        elif "filename" in manifest.columns:
            manifest["stem"] = manifest["filename"].astype(str).map(lambda value: Path(value).stem)
        else:
            raise ValueError(f"{manifest_path} must contain one of: stem, image_id, filename.")

    allowed = {"train", "val", "test"}
    manifest = manifest[["stem", "split"]].copy()
    manifest["stem"] = manifest["stem"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.lower()
    invalid = sorted(set(manifest["split"]) - allowed)
    if invalid:
        raise ValueError(f"{manifest_path} contains unsupported split labels: {invalid}")
    if manifest["stem"].duplicated().any():
        duplicated = manifest.loc[manifest["stem"].duplicated(), "stem"].head(5).tolist()
        raise ValueError(f"{manifest_path} contains duplicate split rows, for example: {duplicated}")

    split_by_stem = manifest.set_index("stem")["split"].to_dict()
    df = image_df.copy()
    df["split"] = df["stem"].map(split_by_stem)
    missing = df.loc[df["split"].isna(), "stem"].head(10).tolist()
    if missing:
        raise ValueError(f"{manifest_path} is missing split assignments for stems: {missing}")
    return df.sort_values("stem").reset_index(drop=True)


def _safe_split(image_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fixed_df = _fixed_split_from_manifest(image_df, config)
    if fixed_df is not None:
        return fixed_df

    split_cfg = config["split"]
    seed = int(split_cfg.get("seed", 42))
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")

    df = image_df.copy()
    class_cols = config["classes"]
    df["dominant_class"] = df[class_cols].idxmax(axis=1)
    try:
        bins = pd.qcut(df["Total"], q=min(4, max(2, df["Total"].nunique())), duplicates="drop")
        strata = df["dominant_class"].astype(str) + "_" + bins.astype(str)
        if strata.value_counts().min() < 2:
            strata = df["dominant_class"].astype(str)
        if strata.value_counts().min() < 2:
            strata = None
    except Exception:
        strata = None

    train_df, holdout_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    holdout_strata = None
    if strata is not None:
        holdout_strata = strata.loc[holdout_df.index]
        if holdout_strata.value_counts().min() < 2:
            holdout_strata = None
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=relative_val,
        random_state=seed + 1,
        shuffle=True,
        stratify=holdout_strata,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("stem").reset_index(drop=True)


def _load_mask(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > threshold


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(max(bw * padding, 1)))
    pad_y = int(round(max(bh * padding, 1)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 - x1 < min_size:
        extra = min_size - (x2 - x1)
        x1 = max(0, x1 - extra // 2)
        x2 = min(width, x2 + extra - extra // 2)
    if y2 - y1 < min_size:
        extra = min_size - (y2 - y1)
        y1 = max(0, y1 - extra // 2)
        y2 = min(height, y2 + extra - extra // 2)
    return x1, y1, x2, y2


def _semantic_from_class_masks(class_masks: dict[str, np.ndarray], classes: list[str]) -> tuple[np.ndarray, int]:
    shape = next(iter(class_masks.values())).shape
    semantic = np.zeros(shape, dtype=np.uint8)
    stacked = np.stack([class_masks[name] for name in classes], axis=0)
    conflict_pixels = int((stacked.sum(axis=0) > 1).sum())
    for class_id, name in enumerate(classes, start=1):
        semantic[class_masks[name]] = class_id
    return semantic, conflict_pixels


def _save_overlay(image: Image.Image, semantic: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB")).copy()
    overlay = rgb.copy()
    for label, color in PALETTE.items():
        if label == 0:
            continue
        mask = semantic == label
        overlay[mask] = np.array(color, dtype=np.uint8)
    blended = (0.65 * rgb + 0.35 * overlay).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(path, quality=92)


def _save_yolo_image(image: Image.Image, target_path: Path, image_size: int) -> tuple[int, int, float, float]:
    width, height = image.size
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(target_path, quality=92)
    return image_size, image_size, image_size / width, image_size / height


def _write_yolo_labels(rows: pd.DataFrame, label_path: Path, width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows.itertuples(index=False):
        x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{int(row.class_index)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _coco_for_split(image_df: pd.DataFrame, instances_df: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    image_id_by_stem = {stem: idx + 1 for idx, stem in enumerate(image_df["stem"].tolist())}
    images = [
        {
            "id": image_id_by_stem[row.stem],
            "file_name": row.filename,
            "path": row.image_path,
            "width": int(row.aligned_width),
            "height": int(row.aligned_height),
        }
        for row in image_df.itertuples(index=False)
    ]
    annotations = []
    for ann_id, row in enumerate(instances_df.itertuples(index=False), start=1):
        width = int(row.x2 - row.x1)
        height = int(row.y2 - row.y1)
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_by_stem[row.stem],
                "category_id": int(row.class_index) + 1,
                "bbox": [float(row.x1), float(row.y1), float(width), float(height)],
                "area": float(row.area),
                "iscrowd": 0,
            }
        )
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
    return {"images": images, "annotations": annotations, "categories": categories}


def _save_audit_plots(image_df: pd.DataFrame, instances_df: pd.DataFrame, dirs: dict[str, Path], classes: list[str]) -> None:
    figures = dirs["audit"] / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    image_df["Total"].hist(bins=24)
    plt.title("Berry Count per Image")
    plt.xlabel("Berries")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(figures / "count_histogram.png", dpi=180)
    plt.close()

    count_by_class = image_df[classes].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    count_by_class.plot(kind="bar", color="#4c78a8")
    plt.title("Workbook Berry Counts by Class")
    plt.ylabel("Berries")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "workbook_class_counts.png", dpi=180)
    plt.close()

    crop_counts = instances_df["class_name"].value_counts().reindex(classes, fill_value=0)
    plt.figure(figsize=(9, 5))
    crop_counts.plot(kind="bar", color="#59a14f")
    plt.title("Extracted Instance/Crop Counts by Class")
    plt.ylabel("Instances")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "extracted_instance_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(image_df["Total"], image_df["class_component_total"], s=18, alpha=0.75)
    lim = max(float(image_df["Total"].max()), float(image_df["class_component_total"].max())) + 2
    plt.plot([0, lim], [0, lim], color="black", linewidth=1)
    plt.title("Workbook Count vs Mask Connected Components")
    plt.xlabel("Workbook total")
    plt.ylabel("Mask component total")
    plt.tight_layout()
    plt.savefig(figures / "count_vs_components.png", dpi=180)
    plt.close()


def prepare_annotations(config: dict[str, Any], rebuild: bool = False) -> PreparedPaths:
    dirs = output_dirs(config)
    data_root = Path(config["data_root"])
    image_root = data_root / config["paths"].get("image_dir", "images")
    classes = list(config["classes"])
    ann_cfg = config.get("annotation", {})
    threshold = int(ann_cfg.get("mask_threshold", 127))
    min_area = int(ann_cfg.get("min_component_area", 64))
    crop_padding = float(ann_cfg.get("crop_padding", 0.18))
    crop_min_size = int(ann_cfg.get("crop_min_size", 20))
    save_overlay_count = int(ann_cfg.get("save_overlay_count", 12))

    image_manifest_path = dirs["annotations"] / "image_manifest.csv"
    instances_path = dirs["annotations"] / "instances.csv"
    crops_path = dirs["annotations"] / "classification_crops.csv"
    audit_report_path = dirs["audit"] / "dataset_audit.md"
    if not rebuild and image_manifest_path.exists() and instances_path.exists() and crops_path.exists():
        return PreparedPaths(image_manifest_path, instances_path, crops_path, audit_report_path)

    extensions = _normalise_exts(ann_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in extensions)
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found in {image_root}")
    image_stems = {path.stem for path in image_paths}
    binary_dir = _binary_mask_dir(data_root, config["paths"].get("binary_mask_aliases", ["ALL", "Overall"]))
    count_workbook = data_root / config["paths"].get("count_workbook", "Image Wise Classname Count.xlsx")
    counts_df, count_audit = _read_counts(count_workbook, classes, image_stems)
    counts_by_stem = counts_df.set_index("stem")

    semantic_dir = dirs["annotations"] / "semantic_masks"
    overlay_dir = dirs["audit"] / "overlays"
    crop_root = dirs["classification"] / "crops"
    yolo_root = dirs["detection"] / "yolo"
    coco_root = dirs["detection"] / "coco"
    for path in [semantic_dir, overlay_dir, crop_root, yolo_root, coco_root]:
        path.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    total_conflict_pixels = 0
    missing_masks: list[str] = []

    for image_idx, image_path in enumerate(tqdm(image_paths, desc="prepare annotations")):
        if image_path.stem not in counts_by_stem.index:
            continue
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
        aligned_width, aligned_height = image.size

        mask_paths = {name: _find_mask(data_root / name, image_path.name) for name in classes}
        binary_mask_path = _find_mask(binary_dir, image_path.name)
        missing = [str(path) for path in list(mask_paths.values()) + [binary_mask_path] if not path.exists()]
        if missing:
            missing_masks.extend(missing)
            continue

        class_masks = {name: _load_mask(path, threshold=threshold) for name, path in mask_paths.items()}
        binary_mask = _load_mask(binary_mask_path, threshold=threshold)
        if binary_mask.shape != (aligned_height, aligned_width):
            raise ValueError(
                f"Mask/image shape mismatch for {image_path.name}: "
                f"mask={binary_mask.shape}, image={(aligned_height, aligned_width)}. "
                "Images must be read with EXIF transpose."
            )

        semantic, conflict_pixels = _semantic_from_class_masks(class_masks, classes)
        total_conflict_pixels += conflict_pixels
        semantic_path = semantic_dir / f"{image_path.stem}.png"
        Image.fromarray(semantic).save(semantic_path)
        if image_idx < save_overlay_count:
            _save_overlay(image, semantic, overlay_dir / f"{image_path.stem}_overlay.jpg")

        counts_row = counts_by_stem.loc[image_path.stem]
        class_component_counts: dict[str, int] = {}
        per_image_instances: list[dict[str, Any]] = []
        for class_index, class_name in enumerate(classes):
            labeled, component_count = ndimage.label(class_masks[class_name])
            kept_for_class = 0
            component_slices = ndimage.find_objects(labeled)
            for component_number, component_slice in enumerate(component_slices, start=1):
                if component_slice is None:
                    continue
                ys, xs = component_slice
                component = labeled[ys, xs] == component_number
                area = int(component.sum())
                if area < min_area:
                    continue
                y1, y2 = int(ys.start), int(ys.stop)
                x1, x2 = int(xs.start), int(xs.stop)
                bbox = (x1, y1, x2, y2)
                kept_for_class += 1
                instance_id = f"{image_path.stem}_{class_name}_{kept_for_class:04d}"
                crop_bbox = _expand_bbox(bbox, aligned_width, aligned_height, crop_padding, crop_min_size)
                crop = image.crop(crop_bbox)
                crop_path = crop_root / class_name / f"{instance_id}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=94)
                row = {
                    "instance_id": instance_id,
                    "stem": image_path.stem,
                    "filename": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "semantic_mask_path": str(semantic_path.resolve()),
                    "source_mask_path": str(mask_paths[class_name].resolve()),
                    "class_name": class_name,
                    "class_index": class_index,
                    "det_label": class_index + 1,
                    "component_number": component_number,
                    "area": area,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_x1": crop_bbox[0],
                    "crop_y1": crop_bbox[1],
                    "crop_x2": crop_bbox[2],
                    "crop_y2": crop_bbox[3],
                    "crop_path": str(crop_path.resolve()),
                }
                instance_rows.append(row)
                per_image_instances.append(row)
                crop_rows.append(
                    {
                        "crop_path": str(crop_path.resolve()),
                        "image_path": str(image_path.resolve()),
                        "stem": image_path.stem,
                        "instance_id": instance_id,
                        "class_name": class_name,
                        "label": class_index,
                    }
                )
            class_component_counts[class_name] = kept_for_class

        overall_labeled, _ = ndimage.label(binary_mask)
        overall_components_kept = 0
        for component_number, component_slice in enumerate(ndimage.find_objects(overall_labeled), start=1):
            if component_slice is None:
                continue
            component = overall_labeled[component_slice] == component_number
            if int(component.sum()) >= min_area:
                overall_components_kept += 1

        row = {
            "stem": image_path.stem,
            "filename": image_path.name,
            "image_path": str(image_path.resolve()),
            "binary_mask_path": str(binary_mask_path.resolve()),
            "semantic_mask_path": str(semantic_path.resolve()),
            "aligned_width": aligned_width,
            "aligned_height": aligned_height,
            "foreground_pixels": int(binary_mask.sum()),
            "class_pixels_total": int(sum(mask.sum() for mask in class_masks.values())),
            "mask_conflict_pixels": conflict_pixels,
            "overall_component_total": int(overall_components_kept),
            "class_component_total": int(sum(class_component_counts.values())),
        }
        for class_name in classes:
            row[class_name] = int(counts_row[class_name])
            row[f"{class_name}_components"] = int(class_component_counts[class_name])
            row[f"{class_name}_pixels"] = int(class_masks[class_name].sum())
        row["Total"] = int(counts_row["Total"])
        row["computed_total"] = int(counts_row["computed_total"])
        image_rows.append(row)

    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} mask files. First missing: {missing_masks[:5]}")

    image_df = pd.DataFrame(image_rows)
    if image_df.empty:
        raise RuntimeError("No images survived annotation preparation.")
    image_df = _safe_split(image_df, config)
    split_by_stem = image_df.set_index("stem")["split"].to_dict()
    instances_df = pd.DataFrame(instance_rows)
    crops_df = pd.DataFrame(crop_rows)
    instances_df["split"] = instances_df["stem"].map(split_by_stem)
    crops_df["split"] = crops_df["stem"].map(split_by_stem)

    image_df.to_csv(image_manifest_path, index=False)
    instances_df.to_csv(instances_path, index=False)
    crops_df.to_csv(crops_path, index=False)
    for split_name in ["train", "val", "test"]:
        image_df[image_df["split"] == split_name].to_csv(dirs["splits"] / f"images_{split_name}.csv", index=False)
        instances_df[instances_df["split"] == split_name].to_csv(dirs["splits"] / f"instances_{split_name}.csv", index=False)
        crops_df[crops_df["split"] == split_name].to_csv(dirs["splits"] / f"crops_{split_name}.csv", index=False)

    _save_audit_plots(image_df, instances_df, dirs, classes)

    yolo_size = int(ann_cfg.get("yolo_image_size", 1024))
    yolo_yaml = yolo_root / "blueberry.yaml"
    for split_name in ["train", "val", "test"]:
        split_images = image_df[image_df["split"] == split_name]
        split_instances = instances_df[instances_df["split"] == split_name]
        split_coco = _coco_for_split(split_images, split_instances, classes)
        with (coco_root / f"instances_{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(split_coco, handle, indent=2)
        for image_row in split_images.itertuples(index=False):
            source_image_path = Path(image_row.image_path)
            with Image.open(source_image_path) as raw_image:
                aligned = ImageOps.exif_transpose(raw_image).convert("RGB")
            yolo_image_path = yolo_root / "images" / split_name / source_image_path.name
            _save_yolo_image(aligned, yolo_image_path, yolo_size)
            label_rows = split_instances[split_instances["stem"] == image_row.stem]
            labels_scaled = label_rows.copy()
            labels_scaled["x1"] = labels_scaled["x1"] * yolo_size / int(image_row.aligned_width)
            labels_scaled["x2"] = labels_scaled["x2"] * yolo_size / int(image_row.aligned_width)
            labels_scaled["y1"] = labels_scaled["y1"] * yolo_size / int(image_row.aligned_height)
            labels_scaled["y2"] = labels_scaled["y2"] * yolo_size / int(image_row.aligned_height)
            _write_yolo_labels(labels_scaled, yolo_root / "labels" / split_name / f"{image_row.stem}.txt", yolo_size, yolo_size)

    yolo_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(classes)}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(classes)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    audit_lines = [
        "# Fresh Blueberry Benchmark Dataset Audit",
        "",
        f"- RGB images prepared: {len(image_df)}",
        f"- Extracted class-mask components / crop candidates: {len(instances_df)}",
        f"- Binary mask folder used: `{binary_dir}`",
        f"- Count workbook rows kept: {count_audit['count_rows_kept']}",
        f"- Workbook rows where `Total` != class sum: {count_audit['workbook_total_mismatch_rows']}",
        f"- Total class-mask overlap/conflict pixels: {total_conflict_pixels}",
        "",
        "## Split Counts",
        "",
        image_df["split"].value_counts().rename_axis("split").reset_index(name="images").to_markdown(index=False),
        "",
        "## Workbook Count Totals",
        "",
        image_df[classes + ["Total"]].sum().to_frame("count").to_markdown(),
        "",
        "## Component Count Agreement",
        "",
        f"- Mean absolute image-level difference, workbook total vs class components: "
        f"{(image_df['Total'] - image_df['class_component_total']).abs().mean():.3f}",
        f"- Max absolute image-level difference: "
        f"{int((image_df['Total'] - image_df['class_component_total']).abs().max())}",
        "",
        "Images with the largest count disagreement:",
        "",
        image_df.assign(abs_diff=(image_df["Total"] - image_df["class_component_total"]).abs())[
            ["filename", "split", "Total", "class_component_total", "overall_component_total", "abs_diff"]
        ]
        .sort_values("abs_diff", ascending=False)
        .head(20)
        .to_markdown(index=False),
        "",
        "## Generated Artifacts",
        "",
        f"- Image manifest: `{image_manifest_path}`",
        f"- Instance boxes/crops: `{instances_path}`",
        f"- Crop classification manifest: `{crops_path}`",
        f"- COCO detection JSON: `{coco_root}`",
        f"- YOLO detection export: `{yolo_root}`",
    ]
    audit_report_path.write_text("\n".join(audit_lines), encoding="utf-8")

    return PreparedPaths(image_manifest_path, instances_path, crops_path, audit_report_path)


def copy_paper_ready_artifacts(config: dict[str, Any]) -> None:
    dirs = output_dirs(config)
    target = dirs["paper_ready"] / "dataset_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    for source in [
        dirs["audit"] / "dataset_audit.md",
        dirs["annotations"] / "image_manifest.csv",
        dirs["annotations"] / "instances.csv",
        dirs["annotations"] / "classification_crops.csv",
    ]:
        if source.exists():
            shutil.copy2(source, target / source.name)
