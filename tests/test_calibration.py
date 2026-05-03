"""Tests for the pixel <-> servo angle calibration store."""

from __future__ import annotations

from pathlib import Path

import pytest

from macbook.calibration import Calibration, CalibrationAnchor, RotationAnchor


class TestCalibration:
    def test_empty_falls_back_to_center(self):
        c = Calibration()
        assert c.pixel_to_servo(640, 360) == (90.0, 90.0)

    def test_custom_center(self):
        c = Calibration(pan_center=45.0, tilt_center=110.0)
        assert c.pixel_to_servo(0, 0) == (45.0, 110.0)

    def test_single_anchor_returned_as_is(self):
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 73.0, 51.0))
        assert c.pixel_to_servo(500, 500) == (73.0, 51.0)

    def test_exact_pixel_hit_returns_anchor_value(self):
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 60, 60))
        c.add(CalibrationAnchor(900, 100, 120, 60))
        c.add(CalibrationAnchor(500, 500, 88, 88))
        pan, tilt = c.pixel_to_servo(500, 500)
        assert pan == 88
        assert tilt == 88

    def test_symmetric_center_interpolates_to_center(self):
        """Five anchors around a center, querying the center returns center."""
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 60, 60))
        c.add(CalibrationAnchor(1180, 100, 120, 60))
        c.add(CalibrationAnchor(640, 360, 90, 90))
        c.add(CalibrationAnchor(100, 620, 60, 120))
        c.add(CalibrationAnchor(1180, 620, 120, 120))
        pan, tilt = c.pixel_to_servo(640, 360)
        assert pan == pytest.approx(90.0, abs=0.5)
        assert tilt == pytest.approx(90.0, abs=0.5)

    def test_clamping_within_servo_limits(self):
        c = Calibration(pan_min=10, pan_max=170, tilt_min=10, tilt_max=170)
        c.add(CalibrationAnchor(100, 100, 200, -50))
        pan, tilt = c.pixel_to_servo(100, 100)
        assert pan == 170
        assert tilt == 10

    def test_remove_nearest_within_radius(self):
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 60, 60))
        c.add(CalibrationAnchor(900, 100, 120, 60))
        assert c.remove_nearest(105, 105) is True
        assert len(c.anchors) == 1
        assert c.anchors[0].pixel_x == 900

    def test_remove_nearest_outside_radius(self):
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 60, 60))
        assert c.remove_nearest(800, 800, max_dist=10) is False
        assert len(c.anchors) == 1

    def test_save_load_round_trip(self, tmp_calibration: Path):
        c = Calibration(frame_width=1920, frame_height=1080, pan_min=20, pan_max=160)
        c.add(CalibrationAnchor(50.0, 60.0, 70.0, 80.0))
        c.add(CalibrationAnchor(1500.0, 1000.0, 130.0, 140.0))
        c.save(tmp_calibration)

        c2 = Calibration.load(tmp_calibration)
        assert c2.frame_width == 1920
        assert c2.pan_min == 20
        assert len(c2.anchors) == 2
        assert c2.anchors[0].pixel_x == 50.0
        assert c2.anchors[1].pan_angle == 130.0

    def test_load_missing_returns_default(self, tmp_path: Path):
        c = Calibration.load(tmp_path / "does_not_exist.json")
        assert c.anchors == []
        assert c.rotation_anchors == []


class TestRotationCalibration:
    def test_empty_returns_rotation_center(self):
        c = Calibration()
        assert c.pixel_to_rotation(640, 360) == 0.0

    def test_custom_rotation_center(self):
        c = Calibration(rotation_center=15.0)
        assert c.pixel_to_rotation(0, 0) == 15.0

    def test_single_rotation_anchor_returned_clamped(self):
        c = Calibration()
        c.add_rotation(RotationAnchor(100, 100, 42.0))
        assert c.pixel_to_rotation(900, 900) == 42.0

    def test_rotation_anchors_independent_of_pan_tilt(self):
        c = Calibration()
        c.add(CalibrationAnchor(100, 100, 60, 60))
        assert c.pixel_to_rotation(100, 100) == 0.0
        assert c.pixel_to_servo(100, 100) == (60.0, 60.0)

    def test_rotation_idw_interpolates(self):
        c = Calibration()
        c.add_rotation(RotationAnchor(0, 360, -90.0))
        c.add_rotation(RotationAnchor(1280, 360, 90.0))
        rot = c.pixel_to_rotation(640, 360)
        assert rot == pytest.approx(0.0, abs=5.0)

    def test_rotation_clamped_to_limits(self):
        c = Calibration(rotation_min=-30.0, rotation_max=30.0)
        c.add_rotation(RotationAnchor(100, 100, 200.0))
        assert c.pixel_to_rotation(100, 100) == 30.0
        c.rotation_anchors.clear()
        c.add_rotation(RotationAnchor(100, 100, -200.0))
        assert c.pixel_to_rotation(100, 100) == -30.0

    def test_rotation_save_load_round_trip(self, tmp_calibration: Path):
        c = Calibration(rotation_min=-90.0, rotation_max=90.0, rotation_center=5.0)
        c.add_rotation(RotationAnchor(100.0, 200.0, -45.0))
        c.add_rotation(RotationAnchor(900.0, 600.0, 30.0))
        c.save(tmp_calibration)

        c2 = Calibration.load(tmp_calibration)
        assert c2.rotation_min == -90.0
        assert c2.rotation_max == 90.0
        assert c2.rotation_center == 5.0
        assert len(c2.rotation_anchors) == 2
        assert c2.rotation_anchors[0].rotation_angle == -45.0
        assert c2.rotation_anchors[1].pixel_x == 900.0

    def test_remove_nearest_rotation(self):
        c = Calibration()
        c.add_rotation(RotationAnchor(100, 100, 10))
        c.add_rotation(RotationAnchor(900, 100, 50))
        assert c.remove_nearest_rotation(105, 105) is True
        assert len(c.rotation_anchors) == 1
        assert c.rotation_anchors[0].pixel_x == 900
