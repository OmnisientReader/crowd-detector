# -*- coding: utf-8 -*-

"""
People (and optionally other classes) detection powered by Ultralytics YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from ultralytics import YOLO


@dataclass
class DetectorConfig:
    """
    Configuration for the detector.

    Attributes:
        model: Path or name of the YOLO weights (e.g. "yolov8n.pt").
        device: "auto" | "cpu" | "cuda" | "mps" (Apple Silicon).
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: Inference image size (int).
        persons_only: If True, filter detections to 'person' class only.
    """
    model: str = "yolov8n.pt"
    device: str = "auto"
    conf: float = 0.30
    iou: float = 0.45
    imgsz: int = 960
    persons_only: bool = True


class PeopleDetector:
    """
    Thin wrapper around Ultralytics YOLO for per-frame inference.

    Usage:
        cfg = DetectorConfig(...)
        det = PeopleDetector(cfg)
        detections = det.predict(frame)
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.model = YOLO(cfg.model)
        self.names = self.model.names  # class id -> name
        self.person_id = self._find_person_class_id(self.names)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve device string from config.

        Returns:
            One of "cuda", "mps" or "cpu".
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _find_person_class_id(names: Dict[int, str]) -> int:
        """Return the class id for 'person' if present, else -1."""
        for k, v in names.items():
            if str(v).lower() == "person":
                return int(k)
        return -1

    def predict(self, frame) -> List[Dict]:
        """
        Run model on a single BGR frame (numpy array) and return detections.

        Returns:
            List of dicts with fields:
                - xyxy: Tuple[int, int, int, int]
                - conf: float
                - cls: int
                - label: str
        """
        classes = None
        if self.cfg.persons_only and self.person_id >= 0:
            classes = [self.person_id]

        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            verbose=False,
            classes=classes,
        )
        r = results[0]
        dets: List[Dict] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cls_id = int(clss[i])
            conf = float(confs[i])
            label = self.names.get(cls_id, str(cls_id))
            dets.append(
                {
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf,
                    "cls": cls_id,
                    "label": str(label),
                }
            )
        return dets
