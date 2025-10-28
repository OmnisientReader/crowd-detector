# -*- coding: utf-8 -*-

"""
Visualization utilities for drawing bounding boxes and labels on frames.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def _put_label(
    img,
    text: str,
    tl: Tuple[int, int],
    font_scale: float = 0.6,
    alpha: float = 0.35,
    color=(0, 255, 0),
    txt_color=(0, 0, 0),
):
    """
    Draw a semi-transparent label background and put text on top.

    Args:
        img: BGR image (numpy array).
        text: Label text to draw.
        tl: Top-left corner for label box.
        font_scale: Font scale for cv2.putText.
        alpha: Transparency of background (0..1).
        color: Background rectangle color (BGR).
        txt_color: Text color (BGR).
    """
    x, y = tl
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    th = th + baseline + 4
    y1 = max(y - th, 0)
    x1 = x
    x2 = x + tw + 6
    y2 = y

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(
        img,
        text,
        (x1 + 3, y2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        txt_color,
        1,
        cv2.LINE_AA,
    )


def draw_detections(
    frame,
    detections: Iterable[Dict],
    line_thickness: int = 2,
    font_scale: float = 0.55,
    alpha: float = 0.35,
):
    """
    Draw bounding boxes and labels for detections on the frame.

    Args:
        frame: BGR image (numpy array).
        detections: Iterable of dicts with keys 'xyxy', 'conf', 'label'.
        line_thickness: Rectangle line thickness.
        font_scale: Label font scale.
        alpha: Label background transparency.

    Returns:
        Annotated frame (same array object is modified in-place).
    """
    if detections is None:
        return frame

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        conf = det["conf"]
        label = det["label"]
        text = f"{label} {conf:.2f}"

        # Light, non-occluding overlay: just the box and a compact label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), line_thickness)

        _put_label(
            frame,
            text=text,
            tl=(x1, y1),
            font_scale=font_scale,
            alpha=alpha,
            color=(144, 238, 144),
            txt_color=(0, 0, 0),
        )

    return frame
