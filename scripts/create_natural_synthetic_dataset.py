from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage
from tqdm import tqdm


CLASS_INFO = [
    {"id": 1, "name": "green_immature", "color_rgb": [35, 170, 70]},
    {"id": 2, "name": "pale_pink", "color_rgb": [255, 150, 190]},
    {"id": 3, "name": "pink_turns_purple", "color_rgb": [155, 70, 210]},
    {"id": 4, "name": "fully_ripe", "color_rgb": [45, 105, 230]},
    {"id": 5, "name": "over_ripe", "color_rgb": [210, 95, 35]},
]

CLASS_NAMES = [item["name"] for item in CLASS_INFO]
CLASS_IDS = {item["name"]: item["id"] for item in CLASS_INFO}
CLASS_COLORS = {item["name"]: tuple(item["color_rgb"]) for item in CLASS_INFO}
TARGET_SYNTHETIC_CLASSES = ["pale_pink", "pink_turns_purple", "fully_ripe", "over_ripe"]


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images: Path
    binary_masks: Path
    semantic_masks: Path
    overall_masks: Path
    metadata: Path
    reports: Path
    figures: Path
    overlays: Path
    splits: Path


@dataclass(frozen=True)
class SourceScene:
    image_id: str
    filename: str
    image_path: str
    overall_mask_path: str
    output_width: int
    output_height: int
    component_count: int


def make_paths(output_root: Path) -> OutputPaths:
    return OutputPaths(
        root=output_root,
        images=output_root / "dataset" / "images",
        binary_masks=output_root / "dataset" / "masks_binary",
        semantic_masks=output_root / "dataset" / "masks_semantic",
        overall_masks=output_root / "dataset" / "masks_overall",
        metadata=output_root / "metadata",
        reports=output_root / "reports",
        figures=output_root / "figures",
        overlays=output_root / "figures" / "overlays",
        splits=output_root / "splits",
    )


def reset_output(paths: OutputPaths, force: bool) -> None:
    if paths.root.exists():
        if not force:
            raise FileExistsError(f"{paths.root} already exists. Pass --force to replace it.")
        shutil.rmtree(paths.root)
    for path in [
        paths.images,
        paths.semantic_masks,
        paths.overall_masks,
        paths.metadata,
        paths.reports,
        paths.figures,
        paths.overlays,
        paths.splits,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        (paths.binary_masks / class_name).mkdir(parents=True, exist_ok=True)


def release_path(release_root: Path, relative_path: str) -> Path:
    return release_root / Path(relative_path.replace("/", "\\"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resized_shape(width: int, height: int, short_side: int) -> tuple[int, int]:
    if width <= height:
        new_width = short_side
        new_height = round(height * short_side / width)
    else:
        new_height = short_side
        new_width = round(width * short_side / height)
    return int(new_width), int(new_height)


def resize_image(image: Image.Image, width: int, height: int, is_mask: bool = False) -> Image.Image:
    return image.resize((width, height), Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS)


def image_quality_metrics(image: Image.Image) -> dict[str, float]:
    small_rgb = image.resize((512, 512), Image.Resampling.LANCZOS)
    gray = np.array(small_rgb.convert("L"), dtype=np.float32)
    laplace = ndimage.laplace(gray)
    hsv = np.array(small_rgb.convert("HSV"), dtype=np.float32)
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "sharpness_laplacian_var": float(laplace.var()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
    }


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {class_name: int(df[f"{class_name}_objects"].sum()) for class_name in CLASS_NAMES}


def collect_target_palettes(
    release_root: Path,
    train_manifest: pd.DataFrame,
    short_side: int,
    max_pixels_per_class: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {class_name: [] for class_name in TARGET_SYNTHETIC_CLASSES}

    for row in tqdm(train_manifest.itertuples(index=False), total=len(train_manifest), desc="sample class colors"):
        image = Image.open(release_path(release_root, row.image_path)).convert("RGB")
        width, height = resized_shape(image.width, image.height, short_side)
        image = resize_image(image, width, height, is_mask=False)
        image_array = np.array(image)
        for class_name in TARGET_SYNTHETIC_CLASSES:
            mask = Image.open(release_path(release_root, getattr(row, f"{class_name}_mask_path"))).convert("L")
            mask = resize_image(mask, width, height, is_mask=True)
            positive = np.array(mask) > 127
            if not positive.any():
                continue
            pixels = image_array[positive]
            if len(pixels) > 300:
                pixels = pixels[rng.choice(len(pixels), size=300, replace=False)]
            samples[class_name].append(pixels.astype(np.float32))

    palettes: dict[str, dict[str, list[float]]] = {}
    fallback = {
        "pale_pink": np.array([175, 132, 142], dtype=np.float32),
        "pink_turns_purple": np.array([116, 78, 129], dtype=np.float32),
        "fully_ripe": np.array([58, 75, 122], dtype=np.float32),
        "over_ripe": np.array([113, 67, 43], dtype=np.float32),
    }
    for class_name in TARGET_SYNTHETIC_CLASSES:
        if samples[class_name]:
            pixels = np.concatenate(samples[class_name], axis=0)
            if len(pixels) > max_pixels_per_class:
                pixels = pixels[rng.choice(len(pixels), size=max_pixels_per_class, replace=False)]
            hsv_pixels = np.array(
                Image.fromarray(pixels.astype(np.uint8).reshape(-1, 1, 3), mode="RGB").convert("HSV"),
                dtype=np.float32,
            ).reshape(-1, 3)
            saturated = hsv_pixels[:, 1] > 18
            stable_hsv = hsv_pixels[saturated] if saturated.any() else hsv_pixels
            mean = np.mean(pixels, axis=0)
            std = np.std(pixels, axis=0)
            median = np.median(pixels, axis=0)
            hsv_mean = np.mean(stable_hsv, axis=0)
            hsv_std = np.std(stable_hsv, axis=0)
            hsv_median = np.median(stable_hsv, axis=0)
        else:
            mean = fallback[class_name]
            std = np.array([20, 20, 20], dtype=np.float32)
            median = fallback[class_name]
            hsv_pixels = np.array(
                Image.fromarray(fallback[class_name].astype(np.uint8).reshape(1, 1, 3), mode="RGB").convert("HSV"),
                dtype=np.float32,
            ).reshape(-1, 3)
            hsv_mean = hsv_pixels[0]
            hsv_std = np.array([8, 18, 22], dtype=np.float32)
            hsv_median = hsv_pixels[0]
        palettes[class_name] = {
            "mean_rgb": mean.round(3).tolist(),
            "std_rgb": std.round(3).tolist(),
            "median_rgb": median.round(3).tolist(),
            "mean_hsv": hsv_mean.round(3).tolist(),
            "std_hsv": hsv_std.round(3).tolist(),
            "median_hsv": hsv_median.round(3).tolist(),
        }
    return palettes


def build_source_scenes(
    release_root: Path,
    train_manifest: pd.DataFrame,
    short_side: int,
) -> list[SourceScene]:
    scenes: list[SourceScene] = []
    for row in tqdm(train_manifest.itertuples(index=False), total=len(train_manifest), desc="index source scenes"):
        image = Image.open(release_path(release_root, row.image_path)).convert("RGB")
        width, height = resized_shape(image.width, image.height, short_side)
        overall = Image.open(release_path(release_root, row.overall_mask_path)).convert("L")
        overall = resize_image(overall, width, height, is_mask=True)
        _, component_count = ndimage.label(np.array(overall) > 127)
        if component_count <= 0:
            continue
        scenes.append(
            SourceScene(
                image_id=row.image_id,
                filename=row.filename,
                image_path=row.image_path,
                overall_mask_path=row.overall_mask_path,
                output_width=width,
                output_height=height,
                component_count=int(component_count),
            )
        )
    if not scenes:
        raise RuntimeError("No source scenes with berry components were found.")
    return scenes


def choose_scene_sequence(scenes: list[SourceScene], target_total: int, seed: int) -> list[SourceScene]:
    counts = sorted({scene.component_count for scene in scenes})
    reachable = [False] * (target_total + 1)
    reachable[0] = True
    for value in range(target_total + 1):
        if not reachable[value]:
            continue
        for count in counts:
            next_value = value + count
            if next_value <= target_total:
                reachable[next_value] = True
    if not reachable[target_total]:
        raise RuntimeError(f"Cannot compose exact target object count {target_total} from source scenes.")

    rng = np.random.default_rng(seed)
    by_count: dict[int, list[SourceScene]] = {}
    for scene in scenes:
        by_count.setdefault(scene.component_count, []).append(scene)

    sequence: list[SourceScene] = []
    remaining = target_total
    reuse_counter: Counter[str] = Counter()
    while remaining > 0:
        valid_counts = [count for count in counts if count <= remaining and reachable[remaining - count]]
        # Prefer moderate object counts and lower source reuse, while keeping the exact DP path possible.
        count_weights = np.array([math.sqrt(count) for count in valid_counts], dtype=float)
        count_weights = count_weights / count_weights.sum()
        chosen_count = int(rng.choice(valid_counts, p=count_weights))
        candidates = by_count[chosen_count]
        reuse_weights = np.array(
            [1.0 / (1.0 + reuse_counter[scene.image_id]) for scene in candidates],
            dtype=float,
        )
        reuse_weights = reuse_weights / reuse_weights.sum()
        scene = candidates[int(rng.choice(len(candidates), p=reuse_weights))]
        sequence.append(scene)
        reuse_counter[scene.image_id] += 1
        remaining -= chosen_count
    return sequence


def assign_component_classes(
    component_count: int,
    deficits: dict[str, int],
    rng: np.random.Generator,
) -> list[str]:
    assignments: list[str] = []
    for _ in range(component_count):
        names = [name for name, value in deficits.items() if value > 0]
        if not names:
            raise RuntimeError("No remaining class deficits to assign.")
        weights = np.array([deficits[name] for name in names], dtype=float)
        weights = weights / weights.sum()
        class_name = str(rng.choice(names, p=weights))
        deficits[class_name] -= 1
        assignments.append(class_name)
    return assignments


def sample_target_rgb(
    class_name: str,
    palettes: dict[str, dict[str, list[float]]],
    rng: np.random.Generator,
) -> np.ndarray:
    palette = palettes[class_name]
    mean = np.array(palette["mean_rgb"], dtype=np.float32)
    std = np.maximum(np.array(palette["std_rgb"], dtype=np.float32), 8.0)
    sampled = rng.normal(mean, std * 0.22)
    class_adjustments = {
        "pale_pink": np.array([12, 8, 10], dtype=np.float32),
        "pink_turns_purple": np.array([4, -2, 12], dtype=np.float32),
        "fully_ripe": np.array([-4, -2, 8], dtype=np.float32),
        "over_ripe": np.array([-2, -8, -10], dtype=np.float32),
    }
    sampled += class_adjustments.get(class_name, 0)
    return np.clip(sampled, 15, 240)


def sample_target_hsv(
    class_name: str,
    palettes: dict[str, dict[str, list[float]]],
    rng: np.random.Generator,
) -> np.ndarray:
    palette = palettes[class_name]
    median = np.array(palette["median_hsv"], dtype=np.float32)
    std = np.maximum(np.array(palette["std_hsv"], dtype=np.float32), np.array([5, 10, 12], dtype=np.float32))
    sampled = median.copy()
    sampled[0] = (rng.normal(median[0], min(std[0] * 0.10, 7.0))) % 256
    sampled[1] = rng.normal(median[1], min(std[1] * 0.18, 18.0))
    sampled[2] = rng.normal(median[2], min(std[2] * 0.14, 18.0))

    limits = {
        "pale_pink": {"s": (18, 92), "v": (82, 210)},
        "pink_turns_purple": {"s": (24, 122), "v": (62, 185)},
        "fully_ripe": {"s": (18, 118), "v": (42, 175)},
        "over_ripe": {"s": (28, 135), "v": (34, 160)},
    }
    class_limits = limits[class_name]
    sampled[1] = np.clip(sampled[1], *class_limits["s"])
    sampled[2] = np.clip(sampled[2], *class_limits["v"])
    return sampled


def recolor_component(
    image_array: np.ndarray,
    component_mask: np.ndarray,
    class_name: str,
    palettes: dict[str, dict[str, list[float]]],
    rng: np.random.Generator,
) -> None:
    y_indices, x_indices = np.where(component_mask)
    if len(y_indices) == 0:
        return
    pad = 4
    y0 = max(0, int(y_indices.min()) - pad)
    y1 = min(component_mask.shape[0], int(y_indices.max()) + pad + 1)
    x0 = max(0, int(x_indices.min()) - pad)
    x1 = min(component_mask.shape[1], int(x_indices.max()) + pad + 1)

    local_mask = component_mask[y0:y1, x0:x1]
    local = image_array[y0:y1, x0:x1, :].astype(np.float32)
    local_hsv = np.array(Image.fromarray(local.astype(np.uint8), mode="RGB").convert("HSV"), dtype=np.float32)
    target_hsv = sample_target_hsv(class_name, palettes, rng)

    inside_hsv = local_hsv[local_mask]
    median_hue = float(np.median(inside_hsv[:, 0]))
    median_sat = float(np.median(inside_hsv[:, 1]))
    median_value = max(float(np.median(inside_hsv[:, 2])), 1.0)

    hue_texture = (local_hsv[:, :, 0] - median_hue) * 0.025
    sat_texture = (local_hsv[:, :, 1] - median_sat) * 0.24
    value_ratio = np.clip(local_hsv[:, :, 2] / median_value, 0.48, 1.62)

    transformed_hsv = local_hsv.copy()
    transformed_hsv[:, :, 0] = (target_hsv[0] + hue_texture + rng.normal(0, 1.2, local_mask.shape)) % 256
    transformed_hsv[:, :, 1] = np.clip(target_hsv[1] + sat_texture + rng.normal(0, 2.0, local_mask.shape), 0, 255)
    transformed_hsv[:, :, 2] = np.clip(target_hsv[2] * np.power(value_ratio, 0.88), 0, 255)
    transformed = np.array(
        Image.fromarray(transformed_hsv.astype(np.uint8), mode="HSV").convert("RGB"),
        dtype=np.float32,
    )

    blurred = ndimage.gaussian_filter(local, sigma=(1.25, 1.25, 0))
    residual = local - blurred
    transformed += residual * 0.52

    inside_values = local_hsv[:, :, 2][local_mask]
    dark_threshold = float(np.percentile(inside_values, 18)) if len(inside_values) else 0.0
    dark_weight = np.zeros(local_mask.shape, dtype=np.float32)
    if dark_threshold > 0:
        dark_weight = np.clip((dark_threshold - local_hsv[:, :, 2]) / dark_threshold, 0.0, 1.0)

    class_strength = {
        "pale_pink": 0.78,
        "pink_turns_purple": 0.82,
        "fully_ripe": 0.76,
        "over_ripe": 0.84,
    }.get(class_name, 0.80)
    alpha = ndimage.gaussian_filter(local_mask.astype(np.float32), sigma=1.05)
    alpha = np.clip(alpha, 0.0, 1.0) * class_strength * (1.0 - dark_weight * 0.42)
    alpha = alpha[:, :, None]
    output = local * (1.0 - alpha) + transformed * alpha
    image_array[y0:y1, x0:x1, :] = np.clip(output, 0, 255).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, format="PNG", optimize=True)


def make_overlay(image: Image.Image, class_masks: dict[str, np.ndarray], output_path: Path, max_width: int = 900) -> None:
    overlay = image.convert("RGBA")
    for class_name in CLASS_NAMES:
        mask = class_masks[class_name]
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask, :3] = CLASS_COLORS[class_name]
        rgba[mask, 3] = 105
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, mode="RGBA"))
    scale = min(1.0, max_width / overlay.width)
    resized = overlay.resize((int(overlay.width * scale), int(overlay.height * scale)), Image.Resampling.LANCZOS)
    strip_h = 24 + 20 * len(CLASS_NAMES)
    canvas = Image.new("RGB", (resized.width, resized.height + strip_h), "white")
    canvas.paste(resized.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = resized.height + 6
    for class_name in CLASS_NAMES:
        draw.rectangle([10, y + 3, 24, y + 17], fill=CLASS_COLORS[class_name])
        draw.text((32, y), f"{class_name}: {int(class_masks[class_name].sum()):,} pixels", fill="black")
        y += 20
    canvas.save(output_path, quality=92)


def make_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 4) -> None:
    if not image_paths:
        return
    thumb_w, thumb_h = 260, 340
    label_h = 26
    pad = 10
    rows = math.ceil(len(image_paths) / columns)
    tile_w = thumb_w + 2 * pad
    tile_h = thumb_h + label_h + 2 * pad
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(image_paths):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + pad + (thumb_w - image.width) // 2, y + pad))
        draw.text((x + pad, y + pad + thumb_h + 4), path.stem, fill="black")
    canvas.save(output_path, quality=92)


def plot_class_balance(table: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(table))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, table["real_train_objects"], width, label="Real train")
    ax.bar(x + width / 2, table["augmented_train_objects"], width, label="Real + natural synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(table["class_name"], rotation=25, ha="right")
    ax.set_ylabel("Object count")
    ax.set_title("Training class balance before and after natural synthetic augmentation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_quality(real_manifest: pd.DataFrame, synthetic_manifest: pd.DataFrame, output_path: Path) -> None:
    metrics = ["brightness_mean", "brightness_std", "sharpness_laplacian_var", "saturation_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.hist(real_manifest[metric], bins=24, alpha=0.55, label="Real", color="#607d8b")
        ax.hist(synthetic_manifest[metric], bins=24, alpha=0.55, label="Natural synthetic", color="#6d8f55")
        ax.set_title(metric)
        ax.set_ylabel("Images")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def imbalance_ratio(counts: dict[str, int]) -> float:
    positive = [value for value in counts.values() if value > 0]
    return max(positive) / min(positive)


def write_reports(
    paths: OutputPaths,
    real_train_counts: dict[str, int],
    synthetic_counts: dict[str, int],
    augmented_counts: dict[str, int],
    synthetic_manifest: pd.DataFrame,
    object_manifest: pd.DataFrame,
    comparison: pd.DataFrame,
    source_reuse: pd.DataFrame,
    palettes: dict[str, dict[str, list[float]]],
) -> None:
    validation = [
        "# Natural Synthetic Augmentation Technical Validation",
        "",
        "## Purpose",
        "",
        "This set replaces random floating copy-paste with instance-preserving ripeness-state simulation. Whole real greenhouse scenes from the training split are reused, existing berry components remain attached to their original stems/branches, and each component is recolored and relabeled into the under-represented ripeness classes.",
        "",
        "Validation and test images are not used as sources and remain real-only for downstream evaluation.",
        "",
        "## Completeness",
        "",
        f"- Natural synthetic images: {len(synthetic_manifest):,}",
        f"- Synthetic relabeled berry components: {len(object_manifest):,}",
        f"- Unique synthetic image hashes: {synthetic_manifest['sha256'].nunique():,}",
        "- Each synthetic image has five binary masks, one overall mask, and one semantic mask.",
        "",
        "## Geometry",
        "",
        synthetic_manifest.groupby(["width", "height"]).size().reset_index(name="images").to_markdown(index=False),
        "",
        "## Class-Balance Result",
        "",
        comparison.to_markdown(index=False),
        "",
        f"- Real training imbalance ratio: {imbalance_ratio(real_train_counts):.2f}:1",
        f"- Augmented training imbalance ratio: {imbalance_ratio(augmented_counts):.2f}:1",
        "",
        "## Mask Integrity",
        "",
        f"- Images with class-mask overlap: {int((synthetic_manifest['overlap_pixels'] > 0).sum())}",
        f"- Maximum overlap pixels: {int(synthetic_manifest['overlap_pixels'].max())}",
        f"- Mean overlap pixels: {synthetic_manifest['overlap_pixels'].mean():.2f}",
        "",
        "## Source Reuse",
        "",
        source_reuse.describe(include="all").to_markdown(),
        "",
        "## Color Simulation Palettes",
        "",
        "Target ripeness colors were sampled from real training masks and used as class-specific HSV/RGB color distributions for texture-preserving recoloring.",
        "",
        "```json",
        json.dumps(palettes, indent=2),
        "```",
        "",
        "## Image Quality Summary",
        "",
        synthetic_manifest[
            ["brightness_mean", "brightness_std", "sharpness_laplacian_var", "saturation_mean"]
        ]
        .describe()
        .reset_index()
        .to_markdown(index=False),
        "",
        "## Validation Figures",
        "",
        "- `figures/class_balance_before_after.png`",
        "- `figures/quality_real_vs_natural_synthetic.png`",
        "- `figures/raw_contact_sheet_natural_synthetic.jpg`",
        "- `figures/overlays/contact_sheet_natural_synthetic_overlays.jpg`",
        "",
    ]
    (paths.reports / "technical_validation_natural_synthetic.md").write_text(
        "\n".join(validation),
        encoding="utf-8",
    )

    report = [
        "# Natural Synthetic Balance Report",
        "",
        "## Main Message",
        "",
        "The first copy-paste synthetic set balanced labels but looked visually artificial because fruit crops were pasted into unconstrained locations. This natural synthetic set keeps the original plant geometry, branch attachment, and greenhouse background by recoloring/relabeling existing berry instances inside real training images.",
        "",
        "## Class Balance",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Reviewer-Safe Use",
        "",
        "- Use this set as training-only augmentation, not as additional real observations.",
        "- Keep the real validation and test splits unchanged.",
        "- In the manuscript, state that synthetic images are included to study imbalance mitigation and rare-class robustness.",
        "- The next proof should be downstream: train real-only vs real+natural-synthetic and test both on the real test set.",
        "",
    ]
    (paths.reports / "natural_synthetic_balance_report.md").write_text("\n".join(report), encoding="utf-8")


def create_natural_synthetic(
    release_root: Path,
    paths: OutputPaths,
    short_side: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    real_manifest = pd.read_csv(release_root / "metadata" / "dataset_manifest.csv")
    train_manifest = real_manifest[real_manifest["split"] == "train"].reset_index(drop=True)
    real_train_counts = class_counts(train_manifest)
    target_per_class = max(real_train_counts.values())
    synthetic_deficits = {
        class_name: max(0, target_per_class - real_train_counts[class_name])
        for class_name in CLASS_NAMES
    }
    synthetic_deficits["green_immature"] = 0
    total_synthetic_objects = sum(synthetic_deficits.values())

    palettes = collect_target_palettes(
        release_root,
        train_manifest,
        short_side=short_side,
        max_pixels_per_class=90000,
        seed=seed,
    )
    scenes = build_source_scenes(release_root, train_manifest, short_side=short_side)
    sequence = choose_scene_sequence(scenes, target_total=total_synthetic_objects, seed=seed + 7)

    pd.DataFrame([scene.__dict__ for scene in scenes]).to_csv(paths.metadata / "source_scene_index.csv", index=False)
    pd.DataFrame([scene.__dict__ for scene in sequence]).to_csv(paths.metadata / "selected_source_sequence.csv", index=False)

    remaining = synthetic_deficits.copy()
    synthetic_rows = []
    object_rows = []
    overlay_paths = []
    raw_sample_paths = []

    for image_index, scene in enumerate(tqdm(sequence, desc="generate natural synthetic"), start=1):
        image_id = f"natural_synthetic_{image_index:05d}"
        filename = f"{image_id}.jpg"
        image = Image.open(release_path(release_root, scene.image_path)).convert("RGB")
        image = resize_image(image, scene.output_width, scene.output_height, is_mask=False)
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.94, 1.06)))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.96, 1.04)))
        image_array = np.array(image)

        overall = Image.open(release_path(release_root, scene.overall_mask_path)).convert("L")
        overall = resize_image(overall, scene.output_width, scene.output_height, is_mask=True)
        overall_mask = np.array(overall) > 127
        labels, component_count = ndimage.label(overall_mask)
        if component_count != scene.component_count:
            raise RuntimeError(f"Component count changed unexpectedly for {scene.filename}")
        assignments = assign_component_classes(component_count, remaining, rng)

        class_masks = {class_name: np.zeros(overall_mask.shape, dtype=bool) for class_name in CLASS_NAMES}
        class_object_counts = {class_name: 0 for class_name in CLASS_NAMES}
        slices = ndimage.find_objects(labels)

        for component_index, component_slice in enumerate(slices, start=1):
            if component_slice is None:
                continue
            component_mask = np.zeros(overall_mask.shape, dtype=bool)
            component_mask[component_slice] = labels[component_slice] == component_index
            class_name = assignments[component_index - 1]
            recolor_component(image_array, component_mask, class_name, palettes, rng)
            class_masks[class_name] |= component_mask
            class_object_counts[class_name] += 1
            y_slice, x_slice = component_slice
            object_rows.append(
                {
                    "synthetic_image_id": image_id,
                    "synthetic_filename": filename,
                    "object_index": component_index,
                    "class_name": class_name,
                    "source_image_id": scene.image_id,
                    "source_filename": scene.filename,
                    "bbox_x": int(x_slice.start),
                    "bbox_y": int(y_slice.start),
                    "bbox_width": int(x_slice.stop - x_slice.start),
                    "bbox_height": int(y_slice.stop - y_slice.start),
                    "mask_pixels": int(component_mask.sum()),
                }
            )

        synthetic_image = Image.fromarray(image_array, mode="RGB")
        image_path = paths.images / filename
        synthetic_image.save(image_path, format="JPEG", quality=93, optimize=True)
        if len(raw_sample_paths) < 16:
            raw_sample_paths.append(image_path)

        semantic = np.zeros(overall_mask.shape, dtype=np.uint8)
        class_sum = np.zeros(overall_mask.shape, dtype=np.uint8)
        for class_name in CLASS_NAMES:
            mask = class_masks[class_name]
            save_binary_mask(mask, paths.binary_masks / class_name / f"{image_id}.png")
            semantic[mask] = CLASS_IDS[class_name]
            class_sum += mask.astype(np.uint8)
        save_binary_mask(overall_mask, paths.overall_masks / f"{image_id}.png")
        Image.fromarray(semantic, mode="L").save(paths.semantic_masks / f"{image_id}.png", format="PNG", optimize=True)

        if len(overlay_paths) < 16:
            overlay_path = paths.overlays / f"{image_id}_overlay.jpg"
            make_overlay(synthetic_image, class_masks, overlay_path)
            overlay_paths.append(overlay_path)

        quality = image_quality_metrics(synthetic_image)
        synthetic_rows.append(
            {
                "image_id": image_id,
                "filename": filename,
                "split": "synthetic_train",
                "source": "natural_ripeness_recolor",
                "source_image_id": scene.image_id,
                "source_filename": scene.filename,
                "image_path": f"dataset/images/{filename}",
                "semantic_mask_path": f"dataset/masks_semantic/{image_id}.png",
                "overall_mask_path": f"dataset/masks_overall/{image_id}.png",
                "width": scene.output_width,
                "height": scene.output_height,
                "total_objects": int(component_count),
                "overlap_pixels": int((class_sum > 1).sum()),
                "sha256": sha256_file(image_path),
                **quality,
                **{f"{class_name}_objects": class_object_counts[class_name] for class_name in CLASS_NAMES},
                **{
                    f"{class_name}_mask_path": f"dataset/masks_binary/{class_name}/{image_id}.png"
                    for class_name in CLASS_NAMES
                },
                **{f"{class_name}_mask_pixels": int(class_masks[class_name].sum()) for class_name in CLASS_NAMES},
            }
        )

    synthetic_manifest = pd.DataFrame(synthetic_rows)
    object_manifest = pd.DataFrame(object_rows)
    synthetic_manifest.to_csv(paths.metadata / "natural_synthetic_manifest.csv", index=False)
    object_manifest.to_csv(paths.metadata / "natural_synthetic_object_manifest.csv", index=False)
    (paths.splits / "natural_synthetic_train.txt").write_text(
        "\n".join(synthetic_manifest["filename"].tolist()) + "\n",
        encoding="utf-8",
    )

    synthetic_counts = class_counts(synthetic_manifest)
    augmented_counts = {
        class_name: real_train_counts[class_name] + synthetic_counts[class_name]
        for class_name in CLASS_NAMES
    }
    comparison = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "real_train_objects": real_train_counts[class_name],
                "natural_synthetic_objects": synthetic_counts[class_name],
                "augmented_train_objects": augmented_counts[class_name],
                "real_train_percent": real_train_counts[class_name] / sum(real_train_counts.values()) * 100.0,
                "augmented_train_percent": augmented_counts[class_name] / sum(augmented_counts.values()) * 100.0,
            }
            for class_name in CLASS_NAMES
        ]
    )
    comparison.to_csv(paths.metadata / "class_balance_comparison.csv", index=False)

    augmented = pd.concat(
        [
            train_manifest.assign(source="real"),
            synthetic_manifest.assign(source="natural_synthetic"),
        ],
        ignore_index=True,
        sort=False,
    )
    augmented.to_csv(paths.metadata / "augmented_train_manifest.csv", index=False)

    source_reuse = (
        synthetic_manifest.groupby(["source_image_id", "source_filename"])
        .size()
        .reset_index(name="reuse_count")
        .sort_values("reuse_count", ascending=False)
    )
    source_reuse.to_csv(paths.metadata / "source_reuse_summary.csv", index=False)

    config = {
        "method": "instance-preserving ripeness-state simulation",
        "source_policy": "real training split only",
        "validation_and_test_policy": "real-only; synthetic data is training-only",
        "short_side": short_side,
        "seed": seed,
        "target_train_objects_per_class": target_per_class,
        "real_train_counts": real_train_counts,
        "natural_synthetic_counts": synthetic_counts,
        "augmented_train_counts": augmented_counts,
        "total_synthetic_objects": int(total_synthetic_objects),
        "synthetic_image_count": int(len(synthetic_manifest)),
        "class_palettes": palettes,
    }
    (paths.metadata / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    plot_class_balance(comparison, paths.figures / "class_balance_before_after.png")
    plot_quality(real_manifest, synthetic_manifest, paths.figures / "quality_real_vs_natural_synthetic.png")
    make_contact_sheet(raw_sample_paths, paths.figures / "raw_contact_sheet_natural_synthetic.jpg")
    make_contact_sheet(overlay_paths, paths.overlays / "contact_sheet_natural_synthetic_overlays.jpg")

    write_reports(
        paths,
        real_train_counts,
        synthetic_counts,
        augmented_counts,
        synthetic_manifest,
        object_manifest,
        comparison,
        source_reuse,
        palettes,
    )

    card = [
        "# Natural Synthetic Blueberry Ripeness Augmentation",
        "",
        "This package contains training-only synthetic images generated by recoloring and relabeling real berry instances in real greenhouse training images. It preserves natural fruit positions and plant attachment while balancing rare ripeness classes.",
        "",
        f"- Natural synthetic images: {len(synthetic_manifest):,}",
        f"- Synthetic relabeled berry components: {len(object_manifest):,}",
        f"- Target augmented training objects per class: {target_per_class:,}",
        "- Method: instance-preserving ripeness-state simulation from real training images.",
        "- Visual validation: raw and mask-overlay contact sheets are provided in `figures/`.",
        "",
        "## Important Use Policy",
        "",
        "Do not count these images as independent real observations. Use them only as training augmentation and report evaluation on real validation/test images.",
        "",
        "## Key Files",
        "",
        "- `metadata/natural_synthetic_manifest.csv`",
        "- `metadata/natural_synthetic_object_manifest.csv`",
        "- `metadata/augmented_train_manifest.csv`",
        "- `metadata/class_balance_comparison.csv`",
        "- `reports/technical_validation_natural_synthetic.md`",
        "- `reports/natural_synthetic_balance_report.md`",
        "",
    ]
    (paths.root / "DATASET_CARD_NATURAL_SYNTHETIC.md").write_text("\n".join(card), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create natural-looking synthetic blueberry ripeness augmentation data.")
    parser.add_argument("--release-root", default="outputs/scientific_data_release")
    parser.add_argument("--output-root", default="outputs/synthetic_balanced_natural_augmentation")
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    release_root = Path(args.release_root).resolve()
    paths = make_paths(Path(args.output_root).resolve())
    reset_output(paths, force=args.force)
    create_natural_synthetic(release_root, paths, short_side=args.short_side, seed=args.seed)
    print(f"Created natural synthetic augmentation at: {paths.root}")
    print(f"Manifest: {paths.metadata / 'natural_synthetic_manifest.csv'}")
    print(f"Validation report: {paths.reports / 'technical_validation_natural_synthetic.md'}")


if __name__ == "__main__":
    main()
