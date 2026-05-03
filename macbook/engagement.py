"""DroneEngagement domain model.

Mirrors the Foundry ontology object type. Used by both the Local JSON
logger and the Foundry OSDK logger so the same record format flows
through both paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class DroneEngagement:
    """One engagement record. Created on detection, mutated as it progresses."""

    drone_id: str
    detection_timestamp: str = field(default_factory=_utcnow_iso)

    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None

    threat_level: ThreatLevel = ThreatLevel.HIGH

    sweep_radius_m: float = 2.0
    engagement_start: str | None = None
    engagement_end: str | None = None
    engagement_duration_s: float | None = None

    fiber_cut: bool = False
    signal_lost: bool = False
    signal_strength: float = 1.0

    sensor_fusion: list[str] = field(default_factory=lambda: ["camera"])
    rf_silence_confirmed: bool = False

    pan_angle: float | None = None
    tilt_angle: float | None = None
    rotation_angle: float | None = None

    notes: str = ""

    def mark_engagement_started(self) -> None:
        self.engagement_start = _utcnow_iso()

    def mark_engagement_ended(self) -> None:
        self.engagement_end = _utcnow_iso()
        if self.engagement_start:
            start = datetime.fromisoformat(self.engagement_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.engagement_end.replace("Z", "+00:00"))
            self.engagement_duration_s = round((end - start).total_seconds(), 3)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["threat_level"] = self.threat_level.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DroneEngagement":
        d = dict(d)
        if "threat_level" in d and isinstance(d["threat_level"], str):
            d["threat_level"] = ThreatLevel(d["threat_level"])
        return cls(**d)
