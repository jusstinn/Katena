"""Tests for the camera overlay rendering.

Mostly smoke tests — we verify it doesn't crash on edge cases (no
detection, signal at extremes, custom resolutions) and that the
output has plausible contents (different from input).
"""

from __future__ import annotations

import numpy as np

from macbook.detector import Detection
from macbook.overlay import (
    SystemStatus,
    draw_crosshair,
    draw_detection,
    draw_signal_bar,
    draw_state_banner,
    draw_status_bar,
    draw_telemetry,
    render_full,
)
from macbook.state_machine import EngagementState, EngagementStateMachine


def _frame(w: int = 1280, h: int = 720) -> np.ndarray:
    return np.full((h, w, 3), 80, dtype=np.uint8)


class TestOverlay:
    def test_state_banner_modifies_pixels(self):
        f = _frame()
        before = f.copy()
        sm = EngagementStateMachine()
        draw_state_banner(f, sm)
        assert not np.array_equal(f, before)

    def test_status_bar_handles_unknown_sdr(self):
        f = _frame()
        draw_status_bar(f, SystemStatus(camera=True, serial=False, foundry=True, sdr=None))

    def test_signal_bar_extremes_do_not_crash(self):
        f = _frame()
        draw_signal_bar(f, 0.0)
        draw_signal_bar(f, 1.0)
        draw_signal_bar(f, -0.5)
        draw_signal_bar(f, 2.0)

    def test_telemetry_with_optional_fields(self):
        f = _frame()
        draw_telemetry(f, fps=30.0)
        draw_telemetry(f, fps=30.0, detection_ms=4.2)
        draw_telemetry(f, fps=30.0, detection_ms=4.2, aim_ms=1.5)

    def test_detection_with_aim_below_frame_clipped(self):
        f = _frame(640, 480)
        det = Detection(bbox=(200, 400, 300, 470), centroid=(250, 435), confidence=0.8, source="motion")
        draw_detection(f, det, state_color=(0, 255, 0), drop_fraction=0.5)

    def test_crosshair_default_center(self):
        f = _frame()
        before = f.copy()
        draw_crosshair(f)
        assert not np.array_equal(f, before)

    def test_render_full_handles_no_detection(self):
        f = _frame()
        render_full(f, EngagementStateMachine(), SystemStatus(), None, fps=0.0, fiber_signal=1.0)

    def test_render_full_full_engagement_render(self):
        f = _frame()
        sm = EngagementStateMachine()
        sm.transition(EngagementState.TARGET_ACQUIRED)
        sm.transition(EngagementState.CLASSIFYING)
        sm.transition(EngagementState.FOG_CONFIRMED)
        sm.transition(EngagementState.ENGAGING)
        det = Detection(bbox=(500, 200, 700, 350), centroid=(600, 275), confidence=0.9, source="ensemble")
        out = render_full(
            f, sm, SystemStatus(camera=True, serial=True, foundry=True, sdr=True),
            det, fps=29.7, detection_ms=4.0, aim_ms=1.0, fiber_signal=0.42,
        )
        assert out.shape == f.shape

    def test_render_full_small_resolution(self):
        f = _frame(320, 240)
        det = Detection(bbox=(50, 50, 150, 200), centroid=(100, 125), confidence=0.6, source="motion")
        render_full(f, EngagementStateMachine(), SystemStatus(camera=True), det, fps=10.0, fiber_signal=0.8)
