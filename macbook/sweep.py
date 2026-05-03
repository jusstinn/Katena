"""Laser sweep trajectory planner.

For the Katena fiber-cut laser we don't aim at a single point — the cable
is ~1mm thin, the drone is moving, and the cable swings. We sweep the
laser through a small ZONE below the drone so the moving cable can't
escape it for long.

Two responsibilities:

  1. Compute the aim ZONE (a centred, possibly rotated rectangle) below
     the drone. The zone trails opposite to the drone's velocity vector
     so a forward-moving drone gets aimed slightly behind it (where the
     fibre actually trails). When the drone is roughly hovering, the
     zone is straight down.

  2. Generate the laser's instantaneous aim POINT at time `t` inside
     that zone, following one of several patterns (lissajous, circle,
     horizontal sweep, etc). All patterns are smooth (sinusoidal) so
     the servos never see step changes — friendly to SG90-class
     hobby servos that hate hard reversals.

Servo-speed safety: every pattern's peak px-velocity is `2π · amplitude
· frequency`. With the defaults this stays well under any reasonable
servo's slew rate, but if you tune amplitudes or frequencies up, check
`peak_speed_px_per_s()` against your servo budget.

Coordinate convention: pixel coordinates in the camera frame, +y down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


SweepPattern = Literal[
    "lissajous", "horizontal", "vertical", "circle", "figure_eight", "static"
]


@dataclass(frozen=True)
class AimZone:
    """A (possibly rotated) rectangle in pixel space.

    `center` is the centre of the rectangle.
    `half_w`, `half_h` are the half-extents BEFORE rotation.
    `angle_rad` rotates the rectangle CCW around the centre. 0 means the
    long axis (height) points straight down (+y).
    """
    center: tuple[float, float]
    half_w: float
    half_h: float
    angle_rad: float

    def corners(self) -> list[tuple[float, float]]:
        """Return the 4 corners in CCW order, after rotation."""
        cx, cy = self.center
        c = math.cos(self.angle_rad)
        s = math.sin(self.angle_rad)
        out = []
        for sx, sy in [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]:
            lx = sx * self.half_w
            ly = sy * self.half_h
            wx = cx + lx * c - ly * s
            wy = cy + lx * s + ly * c
            out.append((wx, wy))
        return out


class SweepPlanner:
    """Compute the aim zone + the current laser aim point inside it.

    Parameters
    ----------
    pattern
        Which trajectory the laser traces inside the zone.
    width_frac, height_frac
        Zone size as a multiple of the drone's bbox width / height.
        Default 0.5 × 1.0 — small, cable-tight.
    amp_x_frac, amp_y_frac
        Sweep amplitude as a fraction of the zone's half-extents. The
        laser's xy oscillates within ±amp×half_extent so the trajectory
        sits inside the zone (not exactly on its boundary).
    freq_x_hz, freq_y_hz
        Pattern frequencies. For lissajous/figure_eight these should be
        non-integer multiples of each other so the curve fills the area
        instead of repeating a small loop. Defaults are 3Hz × 2Hz =
        ratio 1.5, which covers the area cleanly in ~2 sec.
    angle_to_velocity
        If True, rotate the zone so its long axis points OPPOSITE to
        the drone's velocity (cable trails behind). When the drone is
        nearly stationary we revert to straight-down.
    velocity_eps_px_per_s
        Below this speed we treat the drone as hovering — zone is
        straight down regardless of the noisy velocity vector.
    """

    def __init__(
        self,
        pattern: SweepPattern = "lissajous",
        width_frac: float = 0.5,
        height_frac: float = 1.0,
        amp_x_frac: float = 0.7,
        amp_y_frac: float = 0.7,
        # Default frequencies are deliberately LOW (~1Hz). Hobby
        # servos (SG90 / MG90S) and especially heavier MG996Rs cannot
        # cleanly reverse direction much faster than this without
        # visibly stuttering. 1.0Hz x 0.7Hz gives a slow, methodical
        # sweep that the human eye reads as "controlled cutting"
        # rather than "broken jitter".
        freq_x_hz: float = 1.0,
        freq_y_hz: float = 0.7,
        angle_to_velocity: bool = True,
        velocity_eps_px_per_s: float = 60.0,
        # 1.4 puts the zone TOP roughly 0.2 * bbox_height below the
        # drone bbox (clear air gap, no overlap with the airframe).
        # 1.0 = zone top exactly touches drone bottom (no gap).
        # 0.0 = zone centred on the drone.
        zone_drop_frac: float = 1.4,
        # ---- Smoothing (added to kill trail jitter) -----------------
        # YOLO bboxes jump a few px frame-to-frame even on a stationary
        # drone, and the predictor's velocity flips sign on those jumps.
        # Without smoothing the zone CENTRE skips and the zone ROTATION
        # twitches, making the lissajous trail look chunky. EWMA over
        # the bbox centre/size and the velocity vector turns these
        # frame-to-frame jitters into a smooth glide.
        bbox_smoothing_alpha: float = 0.35,
        velocity_smoothing_alpha: float = 0.15,
    ) -> None:
        self.pattern = pattern
        self.width_frac = float(width_frac)
        self.height_frac = float(height_frac)
        self.amp_x_frac = float(amp_x_frac)
        self.amp_y_frac = float(amp_y_frac)
        self.freq_x_hz = float(freq_x_hz)
        self.freq_y_hz = float(freq_y_hz)
        self.angle_to_velocity = bool(angle_to_velocity)
        self.velocity_eps_px_per_s = float(velocity_eps_px_per_s)
        # Where the zone sits relative to the drone bbox. 0.0 = centred
        # on the drone; 1.0 = the zone's TOP touches the drone's bottom
        # (zone fully below). Default 0.6 = mostly-below, slight overlap.
        self.zone_drop_frac = float(zone_drop_frac)
        if not (0.0 < bbox_smoothing_alpha <= 1.0):
            raise ValueError("bbox_smoothing_alpha must be in (0, 1]")
        if not (0.0 < velocity_smoothing_alpha <= 1.0):
            raise ValueError("velocity_smoothing_alpha must be in (0, 1]")
        self.bbox_smoothing_alpha = float(bbox_smoothing_alpha)
        self.velocity_smoothing_alpha = float(velocity_smoothing_alpha)
        self._sm_cx: float | None = None
        self._sm_cy: float | None = None
        self._sm_w: float | None = None
        self._sm_h: float | None = None
        self._sm_vx: float = 0.0
        self._sm_vy: float = 0.0

    # ------------------------------------------------------------------
    # Zone calculation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop smoothing state. Call when the lock is lost or swapped
        so we don't drag stale zone position into the next target."""
        self._sm_cx = None
        self._sm_cy = None
        self._sm_w = None
        self._sm_h = None
        self._sm_vx = 0.0
        self._sm_vy = 0.0

    def zone(
        self,
        bbox: tuple[int, int, int, int],
        velocity_px_per_s: tuple[float, float] = (0.0, 0.0),
    ) -> AimZone:
        x1, y1, x2, y2 = bbox
        raw_bw = max(1, x2 - x1)
        raw_bh = max(1, y2 - y1)
        raw_cx = 0.5 * (x1 + x2)
        raw_cy = 0.5 * (y1 + y2)
        raw_vx, raw_vy = velocity_px_per_s

        # EWMA over the bbox + velocity. First frame seeds the state.
        a_b = self.bbox_smoothing_alpha
        a_v = self.velocity_smoothing_alpha
        if self._sm_cx is None:
            self._sm_cx, self._sm_cy = raw_cx, raw_cy
            self._sm_w, self._sm_h = float(raw_bw), float(raw_bh)
            self._sm_vx, self._sm_vy = raw_vx, raw_vy
        else:
            self._sm_cx = (1 - a_b) * self._sm_cx + a_b * raw_cx
            self._sm_cy = (1 - a_b) * self._sm_cy + a_b * raw_cy
            self._sm_w = (1 - a_b) * self._sm_w + a_b * raw_bw
            self._sm_h = (1 - a_b) * self._sm_h + a_b * raw_bh
            self._sm_vx = (1 - a_v) * self._sm_vx + a_v * raw_vx
            self._sm_vy = (1 - a_v) * self._sm_vy + a_v * raw_vy

        cx = self._sm_cx
        cy = self._sm_cy
        bw = max(1.0, self._sm_w)
        bh = max(1.0, self._sm_h)
        vx = self._sm_vx
        vy = self._sm_vy

        half_w = 0.5 * self.width_frac * bw
        half_h = 0.5 * self.height_frac * bh

        # Default: zone hangs straight down from the drone's belly.
        # The zone's CENTRE is below the bbox centre by (bbox_half_h
        # + zone_half_h * zone_drop_frac).
        offset_along_axis = 0.5 * bh + self.zone_drop_frac * half_h

        speed = math.hypot(vx, vy)
        if self.angle_to_velocity and speed > self.velocity_eps_px_per_s:
            # Cable trails OPPOSITE to motion direction. Velocity
            # vector (vx, vy) points where the drone is going; we want
            # the zone to point the other way.
            # In image coords +y is down, so a stationary cable points
            # along +y -> angle = pi/2 from +x. We rotate CCW.
            # Direction the cable trails (opposite of velocity), as a
            # unit vector:
            tx = -vx / speed
            ty = -vy / speed
            # Place the zone centre along the trail direction.
            zcx = cx + offset_along_axis * tx
            zcy = cy + offset_along_axis * ty
            # Rotate the rectangle so its long (height) axis aligns
            # with the trail direction. Our local "down" in zone-local
            # coords is +y (long axis); we need to map (0,1) -> (tx, ty).
            angle = math.atan2(ty, tx) - math.pi / 2
        else:
            zcx = cx
            zcy = cy + offset_along_axis
            angle = 0.0

        return AimZone(center=(zcx, zcy), half_w=half_w, half_h=half_h,
                       angle_rad=angle)

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def aim_point(self, t: float, zone: AimZone) -> tuple[float, float]:
        """Return the laser's aim pixel at time t inside the zone."""
        ax = zone.half_w * self.amp_x_frac
        ay = zone.half_h * self.amp_y_frac

        # Local coordinates inside the zone (before rotation)
        if self.pattern == "lissajous":
            lx = ax * math.sin(2 * math.pi * self.freq_x_hz * t)
            ly = ay * math.sin(2 * math.pi * self.freq_y_hz * t + math.pi / 2)
        elif self.pattern == "horizontal":
            lx = ax * math.sin(2 * math.pi * self.freq_x_hz * t)
            ly = 0.0
        elif self.pattern == "vertical":
            lx = 0.0
            ly = ay * math.sin(2 * math.pi * self.freq_y_hz * t)
        elif self.pattern == "circle":
            lx = ax * math.cos(2 * math.pi * self.freq_x_hz * t)
            ly = ay * math.sin(2 * math.pi * self.freq_x_hz * t)
        elif self.pattern == "figure_eight":
            # Same x freq, doubled y freq -> classic infinity sign
            lx = ax * math.sin(2 * math.pi * self.freq_x_hz * t)
            ly = ay * math.sin(4 * math.pi * self.freq_x_hz * t) * 0.5
        else:  # "static"
            lx = 0.0
            ly = 0.0

        # Rotate by the zone angle and translate to zone centre
        c = math.cos(zone.angle_rad)
        s = math.sin(zone.angle_rad)
        wx = zone.center[0] + lx * c - ly * s
        wy = zone.center[1] + lx * s + ly * c
        return wx, wy

    def peak_speed_px_per_s(self, zone: AimZone) -> float:
        """Worst-case px/sec the aim point will move. Useful as a
        sanity check vs. the servo's max slew rate."""
        ax = zone.half_w * self.amp_x_frac
        ay = zone.half_h * self.amp_y_frac
        vx_peak = 2 * math.pi * self.freq_x_hz * ax
        vy_peak = 2 * math.pi * self.freq_y_hz * ay
        return math.hypot(vx_peak, vy_peak)


# ---------------------------------------------------------------------
# Rendering helpers (shared by jetson_live_detect.py and replay_video.py)
#
# These live here so both the live and replay scripts produce IDENTICAL
# visualizations -- you tune parameters once in either path and the
# other inherits the same look.
# ---------------------------------------------------------------------

def draw_fire_overlay(
    frame,                                        # np.ndarray
    zone: AimZone | None,
    trail,                                        # deque[(t, (x, y))]
    laser_xy: tuple[float, float] | None,
    *,
    trail_max_age_s: float = 2.0,
    now: float | None = None,
) -> None:
    """Draw the aim zone outline + age-faded laser trail + current dot.

    The trail is rendered from the existing `(t, point)` deque; entries
    older than `trail_max_age_s` are skipped so that during a long
    laser-OFF gap the trail naturally fades out instead of hanging on
    forever (or being wiped instantly).

    All trail segments are drawn onto a SINGLE overlay copy and blended
    once. The age fade is encoded in the per-segment color intensity
    and line thickness, not in the per-segment alpha. This is much
    faster (one frame.copy() + one addWeighted per frame instead of N
    of each) and produces a cleaner visual gradient with no overlapping
    alpha artefacts where adjacent segments share endpoints.

    `zone` and `laser_xy` may be None during cooldown / no-target frames,
    in which case only the trail is drawn -- this is what keeps the
    fading arc visible even after the laser disengages.
    """
    import cv2
    import numpy as np
    import time as _time

    h, w = frame.shape[:2]
    t_now = now if now is not None else _time.perf_counter()

    if zone is not None:
        pts = np.array([[int(round(x)), int(round(y))] for (x, y) in zone.corners()],
                       dtype=np.int32)
        overlay = frame.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255),
                      thickness=1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    items = [(t, p) for (t, p) in trail if (t_now - t) <= trail_max_age_s]
    n = len(items)
    if n >= 2:
        overlay = frame.copy()
        any_drawn = False
        for i in range(1, n):
            (t1, p1) = items[i - 1]
            (t2, p2) = items[i]
            x1i, y1i = int(round(p1[0])), int(round(p1[1]))
            x2i, y2i = int(round(p2[0])), int(round(p2[1]))
            if (x1i < 0 or x1i >= w or y1i < 0 or y1i >= h
                    or x2i < 0 or x2i >= w or y2i < 0 or y2i >= h):
                continue
            seg_age = t_now - 0.5 * (t1 + t2)
            life = max(0.0, 1.0 - seg_age / max(trail_max_age_s, 1e-6))
            r = int(round(255 * life))
            g = int(round(110 * life))
            color = (0, g, r)  # BGR: red core fading to dim
            thickness = 3 if life > 0.75 else (2 if life > 0.30 else 1)
            cv2.line(overlay, (x1i, y1i), (x2i, y2i), color,
                     thickness, cv2.LINE_AA)
            any_drawn = True
        if any_drawn:
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    if laser_xy is not None:
        lx_i, ly_i = int(round(laser_xy[0])), int(round(laser_xy[1]))
        if 0 <= lx_i < w and 0 <= ly_i < h:
            cv2.circle(frame, (lx_i, ly_i), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (lx_i, ly_i), 9, (0, 100, 255), 1, cv2.LINE_AA)


def draw_fire_status(frame, decision, text_fn=None) -> None:
    """Top-right badge for FIRE state. `text_fn` is the script's
    `_shadowed_text` helper (it's defined in each script, not here)."""
    if decision is None or text_fn is None:
        return
    if decision.state == "armed":
        color = (0, 0, 255)
        text = f"\u25CF FIRE  conf={int(decision.conf*100)}%"
    elif decision.state == "cooldown":
        color = (60, 140, 255)
        text = f"COOLDOWN  {decision.dwell_s:0.1f}s"
    else:
        color = (200, 200, 200)
        text = f"TRACK  conf={int(decision.conf*100)}%"
    h, w = frame.shape[:2]
    text_fn(frame, text, (w - 220, 30), scale=0.6, color=color, thickness=2)
