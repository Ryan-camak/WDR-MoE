"""
Visualize YOLO predictions (and optional ground truth) on letterboxed images.

This script is designed for YOLO-Master style projects where you want to:
1) Load a trained YOLO model checkpoint.
2) Run inference on validation images from a dataset YAML.
3) Draw GT + Pred on the same output image.
4) Save visualized outputs to a folder.

Example:
python visualize_yolo_predictions.py \
  --model ./runs/detect/base/weights/best.pt \
  --data ./ribfrac_dataset/meta.yaml \
  --output ./runs/vis/base \
  --imgsz 1024 --conf 0.35 --iou 0.4 --show-label
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import yaml
from ultralytics import YOLO


BBoxDict = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO visualization helper")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--output", type=str, default="runs/vis/ours", help="Output folder for visualized images")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size (if --run-val)")
    parser.add_argument("--run-val", action="store_true", help="Run model.val() before visualization")
    parser.add_argument("--show-label", action="store_true", help="Draw class names/confidence text")
    parser.add_argument(
        "--match-dir",
        type=str,
        default=None,
        help="Only save images whose filename exists in this directory",
    )
    parser.add_argument(
        "--include-gt",
        action="store_true",
        default=True,  # Set default to True
        help="Draw ground-truth boxes from YOLO labels if available",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp",
        help="Comma-separated image extensions to scan",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=40,
        help="Maximum number of images to process from the dataset",
    )
    return parser.parse_args()


def letterbox(
    img,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
) -> Tuple[object, float, Tuple[int, int]]:
    """Resize/pad image to target shape while preserving aspect ratio."""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Optionally pad to stride-multiple for consistent downstream dimensions.
    h, w = img.shape[:2]
    pad_h = int((stride - h % stride) % stride)
    pad_w = int((stride - w % stride) % stride)

    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=color)
        right += pad_w
        bottom += pad_h

    return img, r, (left, top)


def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[BBoxDict]:
    """Load YOLO txt labels and convert to pixel xyxy boxes in original image space."""
    boxes: List[BBoxDict] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Debug: Print parsed label values
            print(f"[DEBUG] Parsed label: class={cls_id}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            boxes.append(
                {
                    "class": cls_id,
                    "bbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                }
            )
    return boxes


def map_boxes_to_letterbox(
    boxes: Sequence[BBoxDict], ratio: float, pad_x: int, pad_y: int, out_w: int, out_h: int
) -> List[BBoxDict]:
    """Map original-image boxes onto the letterboxed image coordinates."""
    mapped: List[BBoxDict] = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]

        x1 = x1 * ratio + pad_x
        y1 = y1 * ratio + pad_y
        x2 = x2 * ratio + pad_x
        y2 = y2 * ratio + pad_y

        x1 = int(round(max(0, min(out_w - 1, x1))))
        y1 = int(round(max(0, min(out_h - 1, y1))))
        x2 = int(round(max(0, min(out_w - 1, x2))))
        y2 = int(round(max(0, min(out_h - 1, y2))))

        item: BBoxDict = {"class": int(box["class"]), "bbox": [x1, y1, x2, y2]}
        if "conf" in box:
            item["conf"] = float(box["conf"])
        mapped.append(item)

    return mapped


def draw_boxes(
    img,
    boxes: Sequence[BBoxDict],
    color: Tuple[int, int, int],
    label_prefix: str,
    class_names: Sequence[str],
    show_label: bool = False,
):
    """Draw rectangles and optional labels on image."""
    img_h, img_w = img.shape[:2]
    scale = max(0.4, min(3.0, ((img_h * img_w) / (1024.0 * 1024.0)) ** 0.5))
    thickness = max(1, int(round(scale * 2)))
    font_scale = max(0.4, scale * 0.6)

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        cls_id = int(box["class"])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if not show_label:
            continue

        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class{cls_id}"
        label = f"{label_prefix} {class_name}"
        if "conf" in box:
            label += f" {float(box['conf']):.2f}"

        (label_w, label_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        y_label = y1 - label_h - int(6 * scale)
        if y_label < 0:
            y_label = y2 + int(6 * scale)

        cv2.rectangle(
            img,
            (x1, y_label),
            (x1 + label_w, y_label + label_h + int(4 * scale)),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, y_label + label_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return img


def resolve_val_dirs(data_yaml: Path) -> Tuple[List[str], Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names.keys())]

    val_entry = data_config.get("val", "val")
    if isinstance(val_entry, list):
        raise ValueError("This script expects a single val path in data yaml.")

    data_root = data_yaml.parent
    val_path = Path(val_entry)

    if not val_path.is_absolute():
        val_images_dir = data_root / val_path
    else:
        val_images_dir = val_path

    # Check if labels are in the same directory as images
    val_labels_dir = val_images_dir
    if not any(val_images_dir.glob("*.txt")):
        # Fallback to 'labels' directory if no labels are found in the image directory
        val_labels_dir = Path(str(val_images_dir).replace("images", "labels"))

    return class_names, val_images_dir, val_labels_dir


def resolve_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    """Resolve the expected YOLO label path for an image.

    Supports:
    - image and txt in the same folder
    - images/* mirrored by labels/*
    - nested subfolders under images/labels
    """
    candidates: List[Path] = []

    # Same-folder labels.
    candidates.append(img_path.with_suffix(".txt"))

    # Mirrored labels folder (preserve relative subpath when possible).
    try:
        rel = img_path.relative_to(images_root)
        candidates.append((labels_root / rel).with_suffix(".txt"))
    except ValueError:
        candidates.append(labels_root / f"{img_path.stem}.txt")

    # Flat labels folder fallback.
    candidates.append(labels_root / f"{img_path.stem}.txt")

    for path in candidates:
        if path.exists():
            return path

    # Return most likely target for easier debugging logs.
    return candidates[1] if len(candidates) > 1 else candidates[0]


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    try:
        stride = int(max(model.model.stride))
    except Exception:
        stride = 32

    if args.run_val:
        print("Running validation metrics...")
        try:
            metrics = model.val(
                imgsz=args.imgsz,
                batch=args.batch,
                iou=args.iou,
                conf=args.conf,
                augment=False,
                data=str(data_yaml),
            )
            print(f"mAP50-95: {metrics.box.map}")
            print(f"mAP50: {metrics.box.map50}")
            print(f"mAP75: {metrics.box.map75}")
        except Exception as exc:
            print(f"Warning: model.val() failed: {exc}")

    class_names, val_images_dir, val_labels_dir = resolve_val_dirs(data_yaml)

    print(f"Validation images: {val_images_dir}")
    print(f"Validation labels: {val_labels_dir}")
    print(f"Output directory: {output_dir}")
    if not args.include_gt:
        print("Note: --include-gt is not set, only predictions will be drawn.")

    match_names: Optional[set[str]] = None
    if args.match_dir:
        match_dir = Path(args.match_dir)
        if match_dir.exists() and match_dir.is_dir():
            match_names = {p.name for p in match_dir.iterdir() if p.is_file()}
            print(f"Match-dir enabled, {len(match_names)} filenames will be considered")
        else:
            print(f"Warning: --match-dir is invalid: {match_dir}. Ignoring filter.")

    image_exts = [ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip()]
    image_files: List[Path] = []
    for ext in image_exts:
        pattern = f"*{ext}" if not ext.startswith(".") else f"*{ext}"
        image_files.extend(val_images_dir.rglob(pattern))

    image_files = sorted(image_files)
    if not image_files:
        raise RuntimeError(f"No images found in {val_images_dir} with extensions={image_exts}")

    if args.max_images is not None:
        image_files = image_files[:args.max_images]

    saved_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")

        if match_names is not None and img_path.name not in match_names:
            skipped_count += 1
            print("  skipped by match-dir")
            continue

        label_path = resolve_label_path(img_path, val_images_dir, val_labels_dir)
        if not label_path.exists():
            print(f"[DEBUG] Label file not found: {label_path}")
        else:
            print(f"[DEBUG] Label file found: {label_path}")

        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"  warning: failed to read image: {img_path}")
            continue

        img_h, img_w = img0.shape[:2]

        img_lb, ratio, (pad_x, pad_y) = letterbox(
            img0.copy(), new_shape=(args.imgsz, args.imgsz), stride=stride
        )
        lb_h, lb_w = img_lb.shape[:2]

        gt_boxes_raw: List[BBoxDict] = []
        if args.include_gt:
            gt_boxes_raw = load_yolo_labels(label_path, img_w, img_h)

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        pred_boxes_raw: List[BBoxDict] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

                x1 = int(round(max(0, min(img_w - 1, x1))))
                y1 = int(round(max(0, min(img_h - 1, y1))))
                x2 = int(round(max(0, min(img_w - 1, x2))))
                y2 = int(round(max(0, min(img_h - 1, y2))))

                pred_boxes_raw.append(
                    {
                        "class": cls_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                    }
                )

        gt_boxes_lb = map_boxes_to_letterbox(gt_boxes_raw, ratio, pad_x, pad_y, lb_w, lb_h)
        pred_boxes_lb = map_boxes_to_letterbox(pred_boxes_raw, ratio, pad_x, pad_y, lb_w, lb_h)

        out = img_lb.copy()
        # Draw Ground Truth boxes (if enabled)
        if args.include_gt:
            out = draw_boxes(out, gt_boxes_lb, (0, 255, 0), "GT", class_names, show_label=args.show_label)

        # Draw Prediction boxes
        out = draw_boxes(out, pred_boxes_lb, (0, 0, 255), "Pred", class_names, show_label=args.show_label)

        # Filter images where GT and predicted box counts match
        if len(gt_boxes_lb) != len(pred_boxes_lb):
            print(f"  skipped: GT count ({len(gt_boxes_lb)}) != Pred count ({len(pred_boxes_lb)})")
            continue

        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), out)
        saved_count += 1

        print(f"  gt={len(gt_boxes_lb)} pred={len(pred_boxes_lb)} -> {save_path}")

    print("-" * 60)
    print(f"Done. total={len(image_files)} skipped={skipped_count} saved={saved_count}")


if __name__ == "__main__":
    main()
