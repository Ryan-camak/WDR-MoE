"""Train WDR-MoE with an Ultralytics-style command line."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-fixwavelet.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to an Ultralytics dataset YAML file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model YAML or checkpoint path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/fracture")
    parser.add_argument("--name", default="wdr_moe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--moe-loss", type=float, default=0.15, dest="moe_loss")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume the latest run or a checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = Path(args.data).expanduser()
    if not data.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data}. "
            "Copy a template from configs/ and update its dataset path first."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        moe=args.moe_loss,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
