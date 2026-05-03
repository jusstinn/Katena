"""Tests for macbook/tracker.py — TargetPredictor + cable_aim_offset."""

from __future__ import annotations

import pytest

from macbook.tracker import TargetPredictor, TrackedTarget, cable_aim_offset


class TestTargetPredictorBasic:
    def test_first_observation_has_zero_velocity(self):
        p = TargetPredictor()
        out = p.update(0.0, 100.0, 200.0)
        assert out.pixel == (100.0, 200.0)
        assert out.velocity == (0.0, 0.0)
        assert out.aim_pixel == (100.0, 200.0)
        assert out.age_frames == 1
        # First obs returns conf=0; we don't trust a single point.
        assert out.confidence == 0.0

    def test_stationary_input_yields_zero_confidence(self):
        """A stationary track is NOT a real target. Confidence should
        collapse via the speed factor (a static blob isn't drone-like).
        """
        p = TargetPredictor(confidence_cap_frames=8)
        for i in range(10):
            out = p.update(i * 0.05, 100.0, 200.0)
        assert out.confidence == pytest.approx(0.0)

    def test_confidence_grows_with_consistent_linear_motion(self):
        """Linearly moving target: conf ramps from low (warmup) toward 1.0."""
        p = TargetPredictor(
            confidence_cap_frames=8,
            speed_full_confidence_px_per_s=200.0,
        )
        confidences: list[float] = []
        for i in range(12):
            # 200 px/s in x, perfectly linear
            out = p.update(i * 0.05, i * 10.0, 100.0)
            confidences.append(out.confidence)
        assert confidences[2] < confidences[7] <= confidences[-1]
        assert confidences[-1] == pytest.approx(1.0, abs=1e-3)

    def test_two_observations_estimate_velocity(self):
        p = TargetPredictor(lead_time_s=0.0)
        p.update(0.0, 100.0, 200.0)
        out = p.update(0.1, 110.0, 200.0)
        # Moved 10 px in x over 0.1 s -> 100 px/s
        assert out.velocity == pytest.approx((100.0, 0.0))
        assert out.aim_pixel == (110.0, 200.0)  # no lead -> aim = current

    def test_lead_time_advances_aim(self):
        p = TargetPredictor(lead_time_s=0.5)
        p.update(0.0, 100.0, 200.0)
        out = p.update(0.1, 110.0, 220.0)
        # vx=100 px/s, vy=200 px/s; lead 0.5s -> +50, +100 from current (110, 220)
        assert out.aim_pixel == pytest.approx((160.0, 320.0))

    def test_age_frames_increments(self):
        p = TargetPredictor()
        for i in range(5):
            out = p.update(i * 0.05, 100.0, 200.0)
        assert out.age_frames == 5

    def test_confidence_caps_at_one_for_clean_linear_track(self):
        p = TargetPredictor(
            confidence_cap_frames=4,
            speed_full_confidence_px_per_s=200.0,
        )
        for i in range(10):
            out = p.update(i * 0.05, i * 10.0, 100.0)
        assert out.confidence == pytest.approx(1.0, abs=1e-3)

    def test_jittering_track_stays_low_confidence(self):
        """Centroid wandering around a fixed point (think: face turning,
        MOG2 noise blob). Speed may look non-zero but trajectory
        straightness collapses to ~0, so confidence stays low."""
        p = TargetPredictor(
            confidence_cap_frames=4,
            speed_full_confidence_px_per_s=200.0,
        )
        # Symmetric jitter ±10 px around (100, 200) -> net displacement ~0
        offsets = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10),
                   (10, 0), (-10, 0), (0, 10), (0, -10), (5, 5)]
        for i, (dx, dy) in enumerate(offsets):
            out = p.update(i * 0.05, 100.0 + dx, 200.0 + dy)
        assert out.confidence < 0.4

    def test_speed_helper_is_l2_norm(self):
        target = TrackedTarget(
            pixel=(0, 0), velocity=(3.0, 4.0),
            aim_pixel=(0, 0), confidence=1.0, age_frames=1,
        )
        assert target.speed_px_per_s == pytest.approx(5.0)


class TestTargetPredictorPrediction:
    def test_constant_velocity_prediction_is_accurate(self):
        """Drone moving at constant 200 px/s in x, predict 0.25s ahead."""
        p = TargetPredictor(lead_time_s=0.25)
        for i in range(5):
            t = i * 0.05  # 50 ms intervals
            x = i * 10.0  # 200 px/s
            p.update(t, x, 100.0)
        out = p.update(0.30, 60.0, 100.0)
        # Last observed x=60 at t=0.3; predict 0.25s ahead at 200 px/s -> 110
        assert out.aim_pixel[0] == pytest.approx(110.0)
        assert out.aim_pixel[1] == pytest.approx(100.0)
        assert out.velocity == pytest.approx((200.0, 0.0))

    def test_diagonal_motion_predicted_in_both_axes(self):
        p = TargetPredictor(lead_time_s=0.1)
        for i in range(4):
            p.update(i * 0.1, i * 50.0, i * 25.0)
        # vx = 50 / 0.1 = 500 px/s, vy = 250 px/s ; current (150, 75) ; lead 0.1 -> (200, 100)
        out = p.update(0.4, 200.0, 100.0)
        assert out.aim_pixel == pytest.approx((250.0, 125.0))


class TestTargetPredictorReset:
    def test_long_gap_resets_track(self):
        p = TargetPredictor(max_gap_s=0.5)
        p.update(0.0, 100.0, 200.0)
        p.update(0.1, 110.0, 200.0)
        # Gap of 1.0s > max_gap_s = 0.5
        out = p.update(1.1, 500.0, 500.0)
        # New track -> single observation, zero velocity
        assert out.velocity == (0.0, 0.0)
        assert out.age_frames == 1

    def test_explicit_reset_clears_history(self):
        p = TargetPredictor()
        p.update(0.0, 100.0, 200.0)
        p.update(0.1, 110.0, 200.0)
        assert p.is_active
        p.reset()
        assert not p.is_active
        out = p.update(0.2, 100.0, 200.0)
        assert out.age_frames == 1
        assert out.velocity == (0.0, 0.0)

    def test_short_gap_keeps_track(self):
        p = TargetPredictor(max_gap_s=0.5)
        p.update(0.0, 100.0, 200.0)
        p.update(0.1, 110.0, 200.0)
        # Gap 0.4s < max_gap_s -> track persists
        out = p.update(0.5, 150.0, 200.0)
        assert out.age_frames == 3


class TestTargetPredictorEdgeCases:
    def test_zero_dt_does_not_divide_by_zero(self):
        p = TargetPredictor()
        p.update(0.0, 100.0, 200.0)
        # Same timestamp -> dt < 1e-3
        out = p.update(0.0, 110.0, 200.0)
        assert out.velocity == (0.0, 0.0)
        assert out.aim_pixel == (110.0, 200.0)

    def test_negative_lead_time_rejected(self):
        with pytest.raises(ValueError):
            TargetPredictor(lead_time_s=-0.1)

    def test_history_too_small_rejected(self):
        with pytest.raises(ValueError):
            TargetPredictor(history=2)

    def test_buffer_caps_at_history_size(self):
        p = TargetPredictor(history=4)
        for i in range(10):
            p.update(i * 0.05, i * 5.0, 100.0)
        # Buffer should hold only N most recent.
        assert len(p._buf) == 4


class TestCableAimOffset:
    def test_zero_offset_is_passthrough(self):
        assert cable_aim_offset((100.0, 200.0)) == (100.0, 200.0)

    def test_y_offset_aims_below_drone(self):
        # Cable hangs DOWN -> +y in image coords
        assert cable_aim_offset((100.0, 200.0), offset_y=40.0) == (100.0, 240.0)

    def test_combined_offset(self):
        # Cable swept slightly left + 30 below
        assert cable_aim_offset((100.0, 200.0), offset_x=-5.0, offset_y=30.0) == (95.0, 230.0)

    def test_returns_floats(self):
        out = cable_aim_offset((100.5, 200.7), offset_y=10.3)
        assert isinstance(out[0], float)
        assert isinstance(out[1], float)
        assert out == (100.5, 211.0)
