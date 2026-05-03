"""Pixel <-> servo angle calibration.

Stores a sparse set of (pixel_x, pixel_y) -> (pan_angle, tilt_angle)
anchor points. Interpolates with inverse-distance-weighting (IDW)
which works on arbitrary anchor configurations (no need for a regular
grid, unlike bilinear).

A second, independent set of anchors maps (pixel_x, pixel_y) -> stepper
rotation in degrees (the 28BYJ-48 base axis that rotates the whole rig
in the XOY plane). Until those rotation anchors are populated,
`pixel_to_rotation` just returns the rotation center, so adding the
stepper to the rig today is a pure no-op for the existing tracker;
calibration data can be added later without touching this code.

The calibration tool (calibration_tool.py) is what populates both
sets. The tracker calls `pixel_to_servo(px, py)` for pan/tilt and
`pixel_to_rotation(px, py)` for the stepper.
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
class RotationAnchor:
    """Pixel -> 28BYJ-48 stepper rotation in degrees."""

    pixel_x: float
    pixel_y: float
    rotation_angle: float


@dataclass
class Calibration:
    anchors: list[CalibrationAnchor] = field(default_factory=list)
    rotation_anchors: list[RotationAnchor] = field(default_factory=list)
    frame_width: int = 1280
    frame_height: int = 720
    pan_min: float = 0.0
    pan_max: float = 180.0
    tilt_min: float = 0.0
    tilt_max: float = 180.0
    pan_center: float = 90.0
    tilt_center: float = 90.0
    rotation_min: float = -180.0
    rotation_max: float = 180.0
    rotation_center: float = 0.0

    def add(self, anchor: CalibrationAnchor) -> None:
        self.anchors.append(anchor)

    def add_rotation(self, anchor: RotationAnchor) -> None:
        self.rotation_anchors.append(anchor)

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

    def remove_nearest_rotation(self, px: float, py: float, max_dist: float = 30.0) -> bool:
        if not self.rotation_anchors:
            return False
        nearest_idx = min(
            range(len(self.rotation_anchors)),
            key=lambda i: self._dist(self.rotation_anchors[i], px, py),
        )
        if self._dist(self.rotation_anchors[nearest_idx], px, py) <= max_dist:
            del self.rotation_anchors[nearest_idx]
            return True
        return False

    @staticmethod
    def _dist(a: CalibrationAnchor | RotationAnchor, px: float, py: float) -> float:
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

    def pixel_to_rotation(
        self, px: float, py: float, k: int = 4, power: float = 2.0
    ) -> float:
        """IDW-interpolate the 28BYJ-48 stepper angle for a given pixel.

        Returns `rotation_center` if no rotation anchors have been
        captured yet — meaning the stepper just stays at zero, and the
        existing pan/tilt servos do all the aiming until you actually
        calibrate the base axis.
        """
        if not self.rotation_anchors:
            return self.rotation_center

        if len(self.rotation_anchors) == 1:
            return self._clamp_rotation(self.rotation_anchors[0].rotation_angle)

        sorted_anchors = sorted(
            self.rotation_anchors, key=lambda a: self._dist(a, px, py)
        )[:k]

        if self._dist(sorted_anchors[0], px, py) < 1e-6:
            return self._clamp_rotation(sorted_anchors[0].rotation_angle)

        weights = [1.0 / (self._dist(a, px, py) ** power) for a in sorted_anchors]
        total = sum(weights)
        rot = sum(w * a.rotation_angle for w, a in zip(weights, sorted_anchors)) / total
        return self._clamp_rotation(rot)

    def _clamp_rotation(self, rot: float) -> float:
        return max(self.rotation_min, min(self.rotation_max, rot))

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
            "rotation_min": self.rotation_min,
            "rotation_max": self.rotation_max,
            "rotation_center": self.rotation_center,
            "anchors": [a.__dict__ for a in self.anchors],
            "rotation_anchors": [a.__dict__ for a in self.rotation_anchors],
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
            rotation_min=d.get("rotation_min", -180.0),
            rotation_max=d.get("rotation_max", 180.0),
            rotation_center=d.get("rotation_center", 0.0),
        )
        for raw in d.get("anchors", []):
            c.anchors.append(CalibrationAnchor(**raw))
        for raw in d.get("rotation_anchors", []):
            c.rotation_anchors.append(RotationAnchor(**raw))
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
