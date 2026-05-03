"""Pixel <-> servo angle calibration.

Stores a sparse set of (pixel_x, pixel_y) -> (pan_angle, tilt_angle)
anchor points. Interpolates with inverse-distance-weighting (IDW)
which works on arbitrary anchor configurations (no need for a regular
grid, unlike bilinear).

The calibration tool (calibration_tool.py) is what populates this.
The tracker just calls `pixel_to_servo(px, py)` to translate detections
into servo commands.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CalibrationAnchor:
    pixel_x: float
    pixel_y: float
    pan_angle: float
    tilt_angle: float


@dataclass
class Calibration:
    anchors: list[CalibrationAnchor] = field(default_factory=list)
    frame_width: int = 1280
    frame_height: int = 720
    pan_min: float = 0.0
    pan_max: float = 180.0
    tilt_min: float = 0.0
    tilt_max: float = 180.0
    pan_center: float = 90.0
    tilt_center: float = 90.0

    def add(self, anchor: CalibrationAnchor) -> None:
        self.anchors.append(anchor)

    def remove_nearest(self, px: float, py: float, max_dist: float = 30.0) -> bool:
        if not self.anchors:
            return False
        nearest_idx = min(
            range(len(self.anchors)),
            key=lambda i: self._dist(self.anchors[i], px, py),
        )
        if self._dist(self.anchors[nearest_idx], px, py) <= max_dist:
            del self.anchors[nearest_idx]
            return True
        return False

    @staticmethod
    def _dist(a: CalibrationAnchor, px: float, py: float) -> float:
        return math.hypot(a.pixel_x - px, a.pixel_y - py)

    def pixel_to_servo(self, px: float, py: float, k: int = 4, power: float = 2.0) -> tuple[float, float]:
        """Inverse-distance-weighted interpolation over the k nearest anchors.

        Falls back to the configured center if no anchors exist.
        Returns (pan_angle, tilt_angle), clamped to configured limits.
        """
        if not self.anchors:
            return self.pan_center, self.tilt_center

        if len(self.anchors) == 1:
            a = self.anchors[0]
            return self._clamp(a.pan_angle, a.tilt_angle)

        # Sort by distance, take k nearest
        sorted_anchors = sorted(self.anchors, key=lambda a: self._dist(a, px, py))[:k]

        # Exact hit
        if self._dist(sorted_anchors[0], px, py) < 1e-6:
            a = sorted_anchors[0]
            return self._clamp(a.pan_angle, a.tilt_angle)

        weights = [1.0 / (self._dist(a, px, py) ** power) for a in sorted_anchors]
        total = sum(weights)
        pan = sum(w * a.pan_angle for w, a in zip(weights, sorted_anchors)) / total
        tilt = sum(w * a.tilt_angle for w, a in zip(weights, sorted_anchors)) / total
        return self._clamp(pan, tilt)

    def _clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        return (
            max(self.pan_min, min(self.pan_max, pan)),
            max(self.tilt_min, min(self.tilt_max, tilt)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "pan_min": self.pan_min,
            "pan_max": self.pan_max,
            "tilt_min": self.tilt_min,
            "tilt_max": self.tilt_max,
            "pan_center": self.pan_center,
            "tilt_center": self.tilt_center,
            "anchors": [a.__dict__ for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibration":
        c = cls(
            frame_width=d.get("frame_width", 1280),
            frame_height=d.get("frame_height", 720),
            pan_min=d.get("pan_min", 0.0),
            pan_max=d.get("pan_max", 180.0),
            tilt_min=d.get("tilt_min", 0.0),
            tilt_max=d.get("tilt_max", 180.0),
            pan_center=d.get("pan_center", 90.0),
            tilt_center=d.get("tilt_center", 90.0),
        )
        for raw in d.get("anchors", []):
            c.anchors.append(CalibrationAnchor(**raw))
        return c

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))
