#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point for running people detection on a video file.

Usage:
    python main.py --input crowd.mp4 --output outputs/crowd_annotated.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
from tqdm.auto import tqdm

from src.detector import PeopleDetector, DetectorConfig
from src.visualization import draw_detections
from src.video_io import (
    open_video_capture,
    create_video_writer,
    ensure_parent_dir,
    video_fps_size,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Detect people on a video and save an annotated copy."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input video file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Path to save the annotated video. "
            "Default: <input_name>_annotated.mp4"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO model weights (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS."
    )
    parser.add_argument(
        "--imgsz", type=int, default=960, help="Inference image size (pixels)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Computation device.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Preview frames in a window while processing (press 'q' to stop).",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Draw all detected classes (not only 'person').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (for quick tests).",
    )
    return parser.parse_args()


def main() -> int:
    """Run detection pipeline."""
    args = parse_args()
    inp_path = Path(args.input)
    if not inp_path.exists():
        print(f"[ERROR] Input file not found: {inp_path}", file=sys.stderr)
        return 2

    out_path = (
        Path(args.output)
        if args.output
        else (inp_path.with_name(f"{inp_path.stem}_annotated.mp4"))
    )
    ensure_parent_dir(out_path)

    cap = open_video_capture(inp_path)
    if not cap or not cap.isOpened():
        print(f"[ERROR] Cannot open video: {inp_path}", file=sys.stderr)
        return 3

    fps, (w, h) = video_fps_size(cap)
    writer = create_video_writer(out_path, fps=fps, size=(w, h))
    if not writer or not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for: {out_path}", file=sys.stderr)
        return 4

    det_cfg = DetectorConfig(
        model=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persons_only=not args.all_classes,
    )
    detector = PeopleDetector(det_cfg)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames:
        total = min(total, args.max_frames)

    pbar = tqdm(total=total or None, desc="Processing", unit="frame")
    shown: bool = False
    processed = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            if args.max_frames and processed > args.max_frames:
                break

            detections = detector.predict(frame)

            # Draw lightweight overlays with class name and confidence
            annotated = draw_detections(
                frame,
                detections,
                line_thickness=2,
                font_scale=0.55,
                alpha=0.35,
            )

            writer.write(annotated)

            if args.show:
                if not shown:
                    cv2.namedWindow("crowd-detector", cv2.WINDOW_NORMAL)
                    shown = True
                cv2.imshow("crowd-detector", annotated)
                # Exit preview by 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
        pbar.close()

    dt = time.time() - t0
    fps_rt = processed / dt if dt > 0 else 0.0
    print(
        f"[INFO] Done. Frames: {processed}, elapsed: {dt:.2f}s, "
        f"avg speed: {fps_rt:.2f} FPS"
    )
    print(f"[INFO] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
