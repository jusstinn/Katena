"""Tests for the DroneEngagement domain model."""

from __future__ import annotations

import time

import pytest

from macbook.engagement import DroneEngagement, ThreatLevel


class TestDroneEngagement:
    def test_minimal_construction(self):
        e = DroneEngagement(drone_id="FOG-001")
        assert e.drone_id == "FOG-001"
        assert e.threat_level == ThreatLevel.HIGH
        assert e.fiber_cut is False
        assert e.signal_lost is False
        assert e.sensor_fusion == ["camera"]

    def test_detection_timestamp_is_utc_iso(self):
        e = DroneEngagement(drone_id="X")
        assert e.detection_timestamp.endswith("Z")
        assert "T" in e.detection_timestamp

    def test_engagement_duration_calculated(self):
        e = DroneEngagement(drone_id="X")
        e.mark_engagement_started()
        time.sleep(0.05)
        e.mark_engagement_ended()
        assert e.engagement_duration_s is not None
        assert 0.04 < e.engagement_duration_s < 0.5

    def test_end_without_start_yields_no_duration(self):
        e = DroneEngagement(drone_id="X")
        e.mark_engagement_ended()
        assert e.engagement_duration_s is None

    def test_dict_round_trip_preserves_all_fields(self):
        e = DroneEngagement(
            drone_id="FOG-007",
            position_x=1.1, position_y=2.2, position_z=3.3,
            threat_level=ThreatLevel.CRITICAL,
            fiber_cut=True,
            signal_lost=True,
            sensor_fusion=["camera", "rf_silence", "ultrasonic"],
            rf_silence_confirmed=True,
            pan_angle=87.5,
            tilt_angle=63.2,
            notes="textbook engagement",
        )
        d = e.to_dict()
        assert d["threat_level"] == "CRITICAL"
        e2 = DroneEngagement.from_dict(d)
        assert e2.drone_id == e.drone_id
        assert e2.threat_level == ThreatLevel.CRITICAL
        assert e2.position_x == 1.1
        assert e2.fiber_cut is True
        assert e2.sensor_fusion == ["camera", "rf_silence", "ultrasonic"]
        assert e2.rf_silence_confirmed is True
        assert e2.pan_angle == 87.5

    @pytest.mark.parametrize("level", list(ThreatLevel))
    def test_all_threat_levels_round_trip(self, level: ThreatLevel):
        e = DroneEngagement(drone_id="X", threat_level=level)
        e2 = DroneEngagement.from_dict(e.to_dict())
        assert e2.threat_level == level
