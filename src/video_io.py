# -*- coding: utf-8 -*-

"""
Video I/O helpers to keep main pipeline clean and cross-platform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import os


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for output file if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def open_video_capture(path: Path) -> cv2.VideoCapture:
    """
    Open a video for reading.

    Returns:
        cv2.VideoCapture instance (opened) or not opened if failure.
    """
    cap = cv2.VideoCapture(str(path))
    return cap


def video_fps_size(cap: cv2.VideoCapture) -> Tuple[float, Tuple[int, int]]:
    """
    Read FPS and frame size from already opened VideoCapture.

    Returns:
        (fps, (width, height))
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return float(fps), (width, height)


def _try_writer(path: Path, fourcc: str, fps: float, size: Tuple[int, int]):
    """Try to open a VideoWriter with given codec."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size
    )
    return writer if writer.isOpened() else None


def create_video_writer(
    path: Path, fps: float, size: Tuple[int, int]
) -> cv2.VideoWriter:
    """
    Create a cross-platform VideoWriter with sensible codec fallbacks.

    Tries: mp4v -> avc1 -> XVID. For .avi suggests XVID.

    Args:
        path: Output file path.
        fps: Frames per second.
        size: (width, height)

    Returns:
        Opened cv2.VideoWriter.
    """
    ext = path.suffix.lower()
    candidates = ["mp4v", "avc1", "XVID"] if ext == ".mp4" else ["XVID", "mp4v"]

    for fourcc in candidates:
        writer = _try_writer(path, fourcc, fps, size)
        if writer:
            return writer

    # Last resort: try system default
    writer = cv2.VideoWriter(str(path), 0, fps, size)
    return writer
