"""Validate a trained WDR-MoE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--data", required=True, help="Path to an Ultralytics dataset YAML file.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="wdr_moe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    from ultralytics import YOLO

    YOLO(str(model_path)).val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
