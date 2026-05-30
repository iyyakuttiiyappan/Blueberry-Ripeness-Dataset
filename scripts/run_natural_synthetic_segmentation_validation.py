from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.config import load_config
from blueberry_multitask.datasets import SegmentationDataset
from blueberry_multitask.engine import _is_better, _loader_kwargs, _optimizer, _segmentation_epoch
from blueberry_multitask.models import create_segmentation_model
from blueberry_multitask.plots import save_history_plot, save_prediction_overlay
from blueberry_multitask.utils import json_dump, now_stamp, resolve_device, set_seed


VARIANTS = ("real_only", "synthetic_only", "real_plus_synthetic")
LOWER_IS_BETTER = {"loss", "val_loss"}


def resolve_manifest_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    output = frame.copy()
    for column in ("image_path", "semantic_mask_path", "overall_mask_path"):
        if column in output.columns:
            output[column] = output[column].map(lambda value: str((root / str(value)).resolve()))
    return output


def load_real_release_manifest(release_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(release_root / "metadata" / "dataset_manifest.csv")
    frame = resolve_manifest_paths(frame, release_root)
    frame["source"] = "real"
    return frame


def load_synthetic_manifest(synthetic_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(synthetic_root / "metadata" / "natural_synthetic_manifest.csv")
    frame = resolve_manifest_paths(frame, synthetic_root)
    frame["split"] = "synthetic_train"
    frame["source"] = "natural_synthetic"
    return frame


def split_frame(frame: pd.DataFrame, split: str, limit: int | None = None) -> pd.DataFrame:
    output = frame[frame["split"] == split].copy()
    if limit is not None:
        output = output.sample(n=min(limit, len(output)), random_state=17)
    return output.reset_index(drop=True)


def class_object_totals(frame: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for class_name in classes:
        column = f"{class_name}_objects"
        totals[class_name] = int(frame[column].sum()) if column in frame.columns else 0
    return totals


def class_pixel_totals(frame: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for class_name in classes:
        columns = [f"{class_name}_mask_pixels", f"{class_name}_pixels"]
        value = 0
        for column in columns:
            if column in frame.columns:
                value = int(frame[column].sum())
                break
        totals[class_name] = value
    return totals


def segmentation_class_weights(
    frame: pd.DataFrame,
    classes: list[str],
    power: float,
    min_weight: float,
    max_weight: float,
) -> tuple[list[float], dict[str, int]]:
    class_pixels = class_pixel_totals(frame, classes)
    total_pixels = int((frame["width"].astype(np.int64) * frame["height"].astype(np.int64)).sum())
    foreground_pixels = int(sum(class_pixels.values()))
    counts = {"background": max(1, total_pixels - foreground_pixels), **class_pixels}
    count_values = np.array([counts["background"], *[counts[name] for name in classes]], dtype=np.float64)
    positive = count_values[count_values > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    weights = np.zeros_like(count_values, dtype=np.float64)
    nonzero = count_values > 0
    weights[nonzero] = np.power(median / count_values[nonzero], power)
    weights[nonzero] = np.clip(weights[nonzero], min_weight, max_weight)
    mean_weight = float(weights[nonzero].mean()) if nonzero.any() else 1.0
    if mean_weight > 0:
        weights[nonzero] = weights[nonzero] / mean_weight
    return weights.astype(float).tolist(), {key: int(value) for key, value in counts.items()}


def make_train_frame(variant: str, real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    real_train = split_frame(real, "train")
    if variant == "real_only":
        return real_train
    if variant == "synthetic_only":
        return synthetic.copy().reset_index(drop=True)
    if variant == "real_plus_synthetic":
        return pd.concat([real_train, synthetic], ignore_index=True, sort=False)
    raise ValueError(f"Unknown variant: {variant}")


def save_variant_metadata(
    run_dir: Path,
    variant: str,
    method: str,
    method_cfg: dict[str, Any],
    created,
    seed: int,
    device: torch.device,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    classes: list[str],
    release_root: Path,
    synthetic_root: Path,
) -> None:
    metadata = {
        "variant": variant,
        "task": "segmentation",
        "method": method,
        "display_name": method_cfg.get("display_name", method),
        "family": method_cfg.get("family", "unknown"),
        "resolved_model_name": created.resolved_name,
        "seed": seed,
        "device": str(device),
        "release_root": str(release_root),
        "synthetic_root": str(synthetic_root),
        "train_images": int(len(train_frame)),
        "val_images": int(len(val_frame)),
        "test_images": int(len(test_frame)),
        "train_sources": train_frame["source"].value_counts().to_dict() if "source" in train_frame else {},
        "train_object_totals": class_object_totals(train_frame, classes),
        "train_pixel_totals": class_pixel_totals(train_frame, classes),
        **created.metadata,
    }
    json_dump(metadata, run_dir / "metadata.json")
    train_frame.to_csv(run_dir / "train_manifest.csv", index=False)
    val_frame.to_csv(run_dir / "val_manifest.csv", index=False)
    test_frame.to_csv(run_dir / "test_manifest.csv", index=False)


def run_variant(
    *,
    config: dict[str, Any],
    method: str,
    variant: str,
    real_manifest: pd.DataFrame,
    synthetic_manifest: pd.DataFrame,
    output_root: Path,
    release_root: Path,
    synthetic_root: Path,
    seed: int,
    device_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    limit: int | None,
    pretrained: bool | None,
    loss_mode: str,
    class_weight_power: float,
    min_class_weight: float,
    max_class_weight: float,
) -> Path:
    classes = list(config["classes"])
    method_cfg = dict(config.get("task_defaults", {}).get("segmentation", {}))
    method_cfg.update(config.get("tasks", {}).get("segmentation", {}).get(method, {}))
    if epochs is not None:
        method_cfg["epochs"] = epochs
    if batch_size is not None:
        method_cfg["batch_size"] = batch_size
    training_cfg = {**config.get("training", {}), **method_cfg}
    if pretrained is not None:
        training_cfg["pretrained"] = pretrained

    set_seed(seed)
    device = resolve_device(device_name)
    image_size = int(method_cfg.get("image_size", 512))
    loader_kwargs = _loader_kwargs(config, device)

    train_frame = make_train_frame(variant, real_manifest, synthetic_manifest)
    val_frame = split_frame(real_manifest, "val", limit)
    test_frame = split_frame(real_manifest, "test", limit)
    if limit is not None:
        train_frame = train_frame.sample(n=min(limit, len(train_frame)), random_state=17).reset_index(drop=True)

    train_loader = DataLoader(
        SegmentationDataset(train_frame, image_size, augment=True),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_frame, image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        SegmentationDataset(test_frame, image_size, augment=False),
        batch_size=int(method_cfg.get("batch_size", 4)),
        shuffle=False,
        **loader_kwargs,
    )

    created = create_segmentation_model(
        method_cfg,
        num_classes=len(classes) + 1,
        pretrained=bool(training_cfg.get("pretrained", True)),
    )
    model = created.model.to(device)
    optimizer = _optimizer(model, training_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(method_cfg.get("epochs", 60))),
    )
    class_weights = None
    class_pixel_counts = None
    if loss_mode == "balanced_ce":
        weights, class_pixel_counts = segmentation_class_weights(
            train_frame,
            classes,
            power=class_weight_power,
            min_weight=min_class_weight,
            max_weight=max_class_weight,
        )
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    run_dir = output_root / "runs" / f"{now_stamp()}_{method}_{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")
    save_variant_metadata(
        run_dir,
        variant,
        method,
        method_cfg,
        created,
        seed,
        device,
        train_frame,
        val_frame,
        test_frame,
        classes,
        release_root,
        synthetic_root,
    )
    if loss_mode == "balanced_ce":
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["loss_mode"] = loss_mode
        metadata["class_weights"] = dict(zip(["background", *classes], [float(value) for value in class_weights.detach().cpu().tolist()]))
        metadata["loss_class_pixel_counts"] = class_pixel_counts
        json_dump(metadata, run_dir / "metadata.json")
    else:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["loss_mode"] = loss_mode
        json_dump(metadata, run_dir / "metadata.json")

    best_value: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    val_metric = str(method_cfg.get("val_metric", "miou_foreground"))
    amp = bool(training_cfg.get("amp", True))
    start = time.perf_counter()

    for epoch in range(1, int(method_cfg.get("epochs", 60)) + 1):
        train_metrics, _, _ = _segmentation_epoch(
            model, train_loader, criterion, optimizer, device, amp, True, len(classes) + 1, classes
        )
        val_metrics, val_per_class, _ = _segmentation_epoch(
            model, val_loader, criterion, optimizer, device, amp, False, len(classes) + 1, classes
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(run_dir / "history.csv", index=False)
        val_per_class.to_csv(run_dir / "per_class_val.csv", index=False)
        save_history_plot(history_frame, run_dir / "training_curves.png", val_metric)
        score = float(val_metrics[val_metric])
        if _is_better(val_metric, score, best_value):
            best_value = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        print(
            f"[{variant}] epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_{val_metric}={score:.4f} best={best_value:.4f}",
            flush=True,
        )
        if bad_epochs >= int(training_cfg.get("early_stopping_patience", 8)):
            print(f"[{variant}] early stopping after {epoch} epochs", flush=True)
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, per_class, sample = _segmentation_epoch(
        model, test_loader, criterion, optimizer, device, amp, False, len(classes) + 1, classes
    )
    per_class.to_csv(run_dir / "per_class_test.csv", index=False)
    if sample is not None:
        image_path, target, pred = sample
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, target, pred, run_dir / "sample_prediction_overlay.jpg")
    test_metrics.update({"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": time.perf_counter() - start})
    json_dump(test_metrics, run_dir / "test_metrics.json")
    print(f"[{variant}] test_metrics={json.dumps(test_metrics, sort_keys=True)}", flush=True)
    return run_dir


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_runs(output_root: Path, run_dirs: list[Path], previous_fresh_summary: Path | None) -> None:
    tables = output_root / "results" / "tables"
    figures = output_root / "results" / "figures"
    reports = output_root / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class_rows = []
    for run_dir in run_dirs:
        metrics = read_json(run_dir / "test_metrics.json")
        metadata = read_json(run_dir / "metadata.json")
        row = {
            "variant": metadata["variant"],
            "method": metadata["method"],
            "display_name": metadata["display_name"],
            "run_dir": str(run_dir),
            "loss_mode": metadata.get("loss_mode", "ce"),
            "train_images": metadata["train_images"],
            "real_train_images": metadata.get("train_sources", {}).get("real", 0),
            "synthetic_train_images": metadata.get("train_sources", {}).get("natural_synthetic", 0),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_metric": metrics.get("best_val_metric"),
            "train_seconds": metrics.get("train_seconds"),
            "miou_foreground": metrics.get("miou_foreground"),
            "dice_foreground": metrics.get("dice_foreground"),
            "pixel_accuracy": metrics.get("pixel_accuracy"),
            "miou": metrics.get("miou"),
            "dice": metrics.get("dice"),
        }
        rows.append(row)
        per_class = pd.read_csv(run_dir / "per_class_test.csv")
        per_class["variant"] = metadata["variant"]
        per_class["run_dir"] = str(run_dir)
        per_class_rows.append(per_class)

    summary = pd.DataFrame(rows)
    order = {variant: idx for idx, variant in enumerate(VARIANTS)}
    summary["variant_order"] = summary["variant"].map(order)
    summary = summary.sort_values("variant_order").drop(columns=["variant_order"])
    summary.to_csv(tables / "segmentation_synthetic_comparison.csv", index=False)
    summary.to_markdown(tables / "segmentation_synthetic_comparison.md", index=False)

    per_class_summary = pd.concat(per_class_rows, ignore_index=True)
    per_class_summary.to_csv(tables / "segmentation_synthetic_per_class.csv", index=False)
    per_class_summary.to_markdown(tables / "segmentation_synthetic_per_class.md", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(summary["variant"], summary["miou_foreground"], color=["#607d8b", "#8e6a99", "#4f8f6f"])
    ax.set_ylabel("Foreground mIoU on real test split")
    ax.set_title("Natural synthetic augmentation segmentation validation")
    ax.set_ylim(0, max(0.05, float(summary["miou_foreground"].max()) * 1.18))
    for index, value in enumerate(summary["miou_foreground"]):
        ax.text(index, float(value) + 0.01, f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "foreground_miou_comparison.png", dpi=220)
    plt.close(fig)

    previous_note = ""
    if previous_fresh_summary is not None and previous_fresh_summary.exists():
        previous = pd.read_csv(previous_fresh_summary)
        previous_seg = previous[previous["task"].eq("segmentation")].copy()
        if not previous_seg.empty:
            previous_seg["miou_foreground"] = pd.to_numeric(previous_seg["miou_foreground"], errors="coerce")
            best = previous_seg.sort_values("miou_foreground", ascending=False).iloc[0]
            previous_note = (
                f"\nExisting fresh-benchmark reference: best real-only segmentation was "
                f"{best['display_name']} with foreground mIoU {float(best['miou_foreground']):.4f}. "
                "This is a reference only because the release split used here is the leak-free split used to generate the synthetic data.\n"
            )

    report = [
        "# Natural Synthetic Segmentation Validation",
        "",
        "## Design",
        "",
        "Three training conditions were evaluated with the same segmentation model, same real validation split, and same real test split.",
        "",
        "- `real_only`: original real training images only.",
        "- `synthetic_only`: natural synthetic images only; this is a domain-transfer sanity check and has no green-immature synthetic labels in the current rare-class augmentation set.",
        "- `real_plus_synthetic`: original real training images plus the natural synthetic rare-class augmentation.",
        "",
        previous_note.strip(),
        "",
        "## Test Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Class Test Results",
        "",
        per_class_summary[["variant", "class_id", "class_name", "iou", "dice"]].to_markdown(index=False),
        "",
        "## Key Outputs",
        "",
        "- `results/tables/segmentation_synthetic_comparison.csv`",
        "- `results/tables/segmentation_synthetic_per_class.csv`",
        "- `results/figures/foreground_miou_comparison.png`",
        "",
    ]
    (reports / "natural_synthetic_segmentation_validation.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate natural synthetic blueberry masks with semantic segmentation.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--release-root", default="outputs/scientific_data_release")
    parser.add_argument("--synthetic-root", default="outputs/synthetic_balanced_natural_augmentation")
    parser.add_argument("--output-root", default="outputs/synthetic_model_validation")
    parser.add_argument("--method", default="fpn_convnextv2_tiny")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--loss-mode", choices=["ce", "balanced_ce"], default="ce")
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--min-class-weight", type=float, default=0.15)
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 42))
    release_root = Path(args.release_root).resolve()
    synthetic_root = Path(args.synthetic_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    real_manifest = load_real_release_manifest(release_root)
    synthetic_manifest = load_synthetic_manifest(synthetic_root)
    run_dirs = []
    for variant in args.variants:
        run_dirs.append(
            run_variant(
                config=config,
                method=args.method,
                variant=variant,
                real_manifest=real_manifest,
                synthetic_manifest=synthetic_manifest,
                output_root=output_root,
                release_root=release_root,
                synthetic_root=synthetic_root,
                seed=seed,
                device_name=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                limit=args.limit,
                pretrained=args.pretrained,
                loss_mode=args.loss_mode,
                class_weight_power=args.class_weight_power,
                min_class_weight=args.min_class_weight,
                max_class_weight=args.max_class_weight,
            )
        )

    summarize_runs(
        output_root,
        run_dirs,
        previous_fresh_summary=Path(config.get("output_root", "outputs/fresh_benchmark"))
        / "results"
        / "tables"
        / "all_task_runs.csv",
    )
    print(f"summary={output_root / 'reports' / 'natural_synthetic_segmentation_validation.md'}")


if __name__ == "__main__":
    main()
