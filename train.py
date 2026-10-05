"""Fine-tune YOLOv8n on the UAV dataset (same settings as configs/train_args.yaml).

The dataset is pulled from Roboflow. Put your own API key in an environment
variable instead of the code:

    export ROBOFLOW_API_KEY=...        # Windows: set ROBOFLOW_API_KEY=...
    python train.py --download --epochs 50

Resume an interrupted run (e.g. a Colab session that timed out):

    python train.py --resume runs/savasan_iha/weights/last.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROBOFLOW_WORKSPACE = "muhammet-hasan-akcesme"
ROBOFLOW_PROJECT = "uavson"
ROBOFLOW_VERSION = 1


def download_dataset() -> Path:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set the ROBOFLOW_API_KEY environment variable first.")
    from roboflow import Roboflow  # imported lazily: only needed for the download

    rf = Roboflow(api_key=key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    return Path(dataset.location) / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the UAV detector")
    p.add_argument("--data", type=Path, default=Path("uavson-1/data.yaml"))
    p.add_argument("--download", action="store_true", help="download the dataset from Roboflow")
    p.add_argument("--model", default="yolov8n.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", type=Path, default=None, help="path to last.pt to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return

    data = download_dataset() if args.download else args.data
    if not data.exists():
        sys.exit(f"Dataset config not found: {data} (use --download)")

    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project="runs", name="savasan_iha", seed=0)
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50:.3f}  mAP@50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
