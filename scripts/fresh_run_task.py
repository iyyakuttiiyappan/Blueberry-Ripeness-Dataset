from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.config import load_config
from blueberry_multitask.engine import run_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fresh multitask benchmark method.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--task", choices=["detection", "segmentation", "counting", "classification"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit images/crops per split for smoke tests.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 42))
    run_dir = run_task(
        config=config,
        task=args.task,
        method=args.method,
        seed=seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
        pretrained=args.pretrained,
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()

