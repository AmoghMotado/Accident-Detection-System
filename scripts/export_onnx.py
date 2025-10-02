#!/usr/bin/env python3
"""
Export a YOLOv8 .pt to ONNX.
Usage:
  python scripts/export_onnx.py --weights weights/yolov8n.pt --out weights/yolov8n.onnx
"""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz, opset=12, half=False, dynamic=True)
    print("ONNX exported; file is in the weights directory with .onnx extension (ultralytics naming).")

if __name__ == "__main__":
    main()
