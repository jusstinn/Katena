"""Drone detection — OpenCV motion + YOLOv8 ensemble.

Two parallel branches running on every frame:

  Motion branch (always on, primary, reliable):
      MOG2 background subtractor -> morphology -> largest contour
      Detects ANY moving object large enough. Robust against poor
      lighting and works without any training.

  YOLO branch (optional, cosmetic + ensemble):
      yolov8n.pt (or fine-tuned weights). Pretrained COCO has no
      "drone" class so out of the box it sees person/bottle/etc.
      We accept any YOLO detection that overlaps the motion bbox to
      boost confidence. After fine-tuning on the kids' drone, this
      branch becomes the primary classifier.

The detector returns at most one Detection per frame: the best
candidate by an ensemble score. The aim_point property gives the
servo target — the brief's "where the fiber exits under the motors,"
i.e. bottom-center of bbox offset slightly below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    """A single drone detection with everything the tracker needs."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    centroid: tuple[int, int]
    confidence: float
    source: str  # "motion", "yolo", or "ensemble"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def aim_point(self, drop_fraction: float = 0.15) -> tuple[int, int]:
        """Where the servos should aim — fiber exit zone under the motors.

        Bottom-center of bbox, offset DOWN by `drop_fraction` of the
        bbox height. Default 15 %: high enough that the laser dot lands
        in the fiber-spool zone of a typical FPV drone, low enough that
        it won't aim above the airframe at small bbox sizes.
        """
        x1, _, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        ay = y2 + int(drop_fraction * self.height)
        return cx, ay


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class MotionDetector:
    """Background-subtraction motion detector. Picks the largest moving blob."""

    def __init__(
        self,
        min_area: int = 400,
        max_area: int = 200_000,
        var_threshold: int = 25,
        history: int = 200,
        learning_rate: float = -1.0,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> Detection | None:
        mask = self._bg.apply(frame, learningRate=self.learning_rate)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            if area > best_area:
                x, y, w, h = cv2.boundingRect(c)
                best = (x, y, x + w, y + h)
                best_area = area
        if best is None:
            return None

        x1, y1, x2, y2 = best
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        confidence = min(1.0, best_area / 10_000.0)
        return Detection(
            bbox=best, centroid=(cx, cy), confidence=confidence, source="motion"
        )


class YoloDetector:
    """YOLOv8 wrapper. Returns the highest-confidence detection on the frame."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        target_classes: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.device = device

    def detect(self, frame: np.ndarray) -> Detection | None:
        kwargs: dict[str, Any] = {"verbose": False, "conf": self.conf_threshold}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = -1
        best_conf = 0.0
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            if self.target_classes and cls_name not in self.target_classes:
                continue
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        if best_idx < 0:
            return None

        box = boxes[best_idx]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        return Detection(
            bbox=(x1, y1, x2, y2),
            centroid=((x1 + x2) // 2, (y1 + y2) // 2),
            confidence=best_conf,
            source="yolo",
        )


class EnsembleDetector:
    """Runs motion + (optional) YOLO. Combines results sensibly.

    Decision logic:
      - If both fire and bboxes overlap (IoU > 0.2) → ensemble (high confidence)
      - If only YOLO fires above its threshold → use YOLO
      - If only motion fires → use motion
      - If both fire but disagree → prefer motion (more reliable on small drones)
    """

    def __init__(
        self,
        motion: MotionDetector | None = None,
        yolo: YoloDetector | None = None,
        iou_match: float = 0.2,
    ) -> None:
        self.motion = motion or MotionDetector()
        self.yolo = yolo
        self.iou_match = iou_match

    def detect(self, frame: np.ndarray) -> Detection | None:
        motion_det = self.motion.detect(frame)
        yolo_det = self.yolo.detect(frame) if self.yolo else None

        if motion_det and yolo_det:
            iou = _bbox_iou(motion_det.bbox, yolo_det.bbox)
            if iou >= self.iou_match:
                box = motion_det.bbox
                conf = min(1.0, (motion_det.confidence + yolo_det.confidence) / 2.0 + 0.2)
                return Detection(
                    bbox=box,
                    centroid=motion_det.centroid,
                    confidence=conf,
                    source="ensemble",
                )
            return motion_det
        if yolo_det:
            return yolo_det
        return motion_det
