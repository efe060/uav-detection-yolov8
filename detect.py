"""Run the trained UAV detector on an image, a folder, a video file or a webcam.

Examples
--------
    python detect.py --source samples/sky.jpg
    python detect.py --source flight.mp4 --conf 0.35
    python detect.py --source 0            # webcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("weights/best.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UAV detection with a fine-tuned YOLOv8n model")
    p.add_argument("--source", required=True,
                   help="image, folder, video file, URL or webcam index (e.g. 0)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"model weights (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--device", default=None, help="'cpu', '0' for the first GPU, ...")
    p.add_argument("--show", action="store_true", help="show a live preview window")
    p.add_argument("--no-save", action="store_true", help="do not write annotated output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        sys.exit(f"Weights not found: {args.weights}\n"
                 "Download best.pt from the Releases page into weights/, "
                 "or train your own model with train.py.")

    model = YOLO(str(args.weights))
    source = int(args.source) if args.source.isdigit() else args.source

    total = 0
    for result in model.predict(source=source, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, stream=True, show=args.show,
                                save=not args.no_save, verbose=False):
        boxes = result.boxes
        total += len(boxes)
        name = Path(result.path).name
        if len(boxes):
            best = float(boxes.conf.max())
            print(f"{name}: {len(boxes)} detection(s), best confidence {best:.2f}")

    print(f"Done. {total} detection(s) in total.")
    if not args.no_save:
        print("Annotated output saved under runs/detect/")


if __name__ == "__main__":
    main()
