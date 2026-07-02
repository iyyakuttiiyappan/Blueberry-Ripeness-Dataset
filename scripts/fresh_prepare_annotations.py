from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.annotations import copy_paper_ready_artifacts, prepare_annotations
from blueberry_multitask.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RGB-image + mask annotations for the fresh benchmark.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = prepare_annotations(config, rebuild=args.rebuild)
    copy_paper_ready_artifacts(config)
    print(f"image_manifest={paths.image_manifest}")
    print(f"instances={paths.instances}")
    print(f"crops={paths.crops}")
    print(f"audit_report={paths.audit_report}")


if __name__ == "__main__":
    main()

