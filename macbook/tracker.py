"""Target tracking + motion prediction for the BlackFiber pipeline.

A `TargetPredictor` consumes per-frame `(timestamp, x, y)` observations
of the drone (typically the bbox centroid from the detector) and
returns a `TrackedTarget` with:

  * a smoothed current pixel position
  * an estimated pixel-space velocity (vx, vy in px/sec)
  * a predicted future pixel position offset by `lead_time_s`
  * a `confidence` that grows as the track stabilises

This is the leading-the-target logic. With a real laser there's a
detection-to-laser-on latency budget of roughly 200-400 ms (mostly
servo movement time). A drone moving 3 m/s for 300 ms travels ~1 m,
which translates to a 50-100 px miss at typical demo distances. So
we aim at *where the drone will be*, not where it is now.

The predictor is intentionally simple — least-squares-style linear
extrapolation over the last few detections. A Kalman filter would
be slightly smoother but adds tuning surface; for the hackathon
this is the right complexity level.

Design notes:
  * Detection-frame ordered. Out-of-order or duplicate timestamps
    are tolerated but degrade prediction quality.
  * If no detection arrives for more than `max_gap_s`, the track is
    considered lost and history is cleared. The next detection
    starts a fresh track.
  * `lead_time_s = 0.0` disables prediction (returns the raw point).
    Use this until you've measured your actual servo latency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


def _trajectory_straightness(points: list[tuple[float, float]]) -> float:
    """Net displacement over total path length. 1.0 = perfectly linear.

    Drones move (mostly) in straight lines. A jittering centroid of a
    head-turn or noise blob produces near-zero net displacement no
    matter how big the per-step jumps look. This is a stronger
    "is-this-really-moving-through-space" signal than raw speed alone.
    """
    if len(points) < 3:
        return 0.0
    net = math.hypot(points[-1][0] - points[0][0],
                     points[-1][1] - points[0][1])
    path = 0.0
    for i in range(1, len(points)):
        path += math.hypot(points[i][0] - points[i - 1][0],
                           points[i][1] - points[i - 1][1])
    return net / path if path > 1e-3 else 0.0


@dataclass(frozen=True)
class TrackedTarget:
    """One frame of tracking output for the state machine."""

    pixel: tuple[float, float]
    """Current estimated drone position (px), latest observation."""

    velocity: tuple[float, float]
    """Pixel-space velocity in (vx, vy) px/sec. Zero if track is fresh."""

    aim_pixel: tuple[float, float]
    """Where to actually aim — current pixel plus lead × velocity.

    NOTE: this does NOT yet include the cable-cut offset. The state
    machine / aim policy is responsible for adding that downstream
    so the offset can vary per-mode (e.g. zero for tracking-only,
    fixed pixels-below for FIRE).
    """

    confidence: float
    """Track confidence in [0, 1].

    Composite of three signals so it actually MEANS something instead of
    saturating to 1.0 after a few frames:

      * age_factor    — ramps over the first few detections (warmup)
      * speed_factor  — 0 when stationary, 1 when moving > ~thresh
      * straightness  — 1 if trajectory is linear, 0 if jittering in place

    A real drone in linear flight stabilises near 1.0. A face turning
    that briefly enters the track stays low (straightness ≈ 0). A
    fresh just-acquired track ramps from low to high over ~8 frames."""

    age_frames: int
    """How many consecutive detections have been fed to this track."""

    @property
    def speed_px_per_s(self) -> float:
        vx, vy = self.velocity
        return (vx * vx + vy * vy) ** 0.5


def cable_aim_offset(
    pixel: tuple[float, float],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[float, float]:
    """Apply the fiber-cable aim offset.

    For the BlackFiber neutralisation strategy we don't aim AT the drone,
    we aim at the trailing fiber-optic tether. The cable hangs roughly
    straight down from the airframe, so a positive `offset_y` (pixels
    BELOW the drone) is usually what you want. Tune empirically once
    the rig is calibrated.

    Returns the offset pixel as floats so downstream calibration math
    keeps sub-pixel resolution. Cast to int only at draw / send time.
    """
    px, py = pixel
    return px + offset_x, py + offset_y


class TargetPredictor:
    """Linear motion predictor with stale-track reset.

    Parameters
    ----------
    lead_time_s
        How far in the future (seconds) to predict the drone's
        position when generating `aim_pixel`. 0.0 disables prediction.
    history
        Maximum recent detections to retain for velocity estimation.
        Larger = smoother but laggier.
    max_gap_s
        If no detection arrives for this many seconds, drop the track
        and start fresh on the next detection.
    confidence_cap_frames
        Number of consecutive detections at which `confidence` saturates
        to 1.0. Earlier frames return a proportional confidence.
    """

    def __init__(
        self,
        lead_time_s: float = 0.0,
        history: int = 8,
        max_gap_s: float = 0.5,
        confidence_cap_frames: int = 8,
        speed_full_confidence_px_per_s: float = 200.0,
    ) -> None:
        if lead_time_s < 0:
            raise ValueError("lead_time_s must be >= 0")
        if history < 3:
            raise ValueError("history must be >= 3 for straightness")
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be > 0")
        if confidence_cap_frames < 1:
            raise ValueError("confidence_cap_frames must be >= 1")

        self.lead_time_s = lead_time_s
        self.max_gap_s = max_gap_s
        self.confidence_cap_frames = confidence_cap_frames
        self.speed_full_confidence_px_per_s = float(speed_full_confidence_px_per_s)
        self._buf: deque[tuple[float, float, float]] = deque(maxlen=history)
        self._frames_tracked = 0

    def reset(self) -> None:
        """Drop the current track. Call when target is lost or scene changes."""
        self._buf.clear()
        self._frames_tracked = 0

    @property
    def is_active(self) -> bool:
        """True if a track is currently being followed."""
        return len(self._buf) > 0

    @property
    def age_frames(self) -> int:
        return self._frames_tracked

    def update(self, t: float, x: float, y: float) -> TrackedTarget:
        """Feed one detection and return the corresponding aim target.

        Parameters
        ----------
        t
            Timestamp of the detection (seconds, any monotonic clock).
        x, y
            Pixel coordinates of the drone (typically the bbox centroid).
        """
        # Reset if too long since last observation -> stale track.
        if self._buf and (t - self._buf[-1][0]) > self.max_gap_s:
            self.reset()

        self._buf.append((t, x, y))
        self._frames_tracked += 1

        # Single observation: no velocity yet.
        if len(self._buf) < 2:
            return TrackedTarget(
                pixel=(x, y),
                velocity=(0.0, 0.0),
                aim_pixel=(x, y),
                confidence=0.0,
                age_frames=self._frames_tracked,
            )

        # Linear velocity from oldest-to-newest in the buffer.
        # (Simple and robust; a least-squares fit doesn't help much
        # at history sizes of 3-7.)
        t0, x0, y0 = self._buf[0]
        tn, xn, yn = self._buf[-1]
        dt = tn - t0
        if dt < 1e-3:
            vx = vy = 0.0
        else:
            vx = (xn - x0) / dt
            vy = (yn - y0) / dt

        ax = xn + vx * self.lead_time_s
        ay = yn + vy * self.lead_time_s

        # --- Composite confidence ---------------------------------------
        # Each factor is in [0, 1]. We use a product so any one being
        # near-zero (no motion / wandering trajectory / fresh track)
        # collapses confidence — mirrors physical "I am sure this is a
        # real linearly-moving target".
        age_factor = min(1.0, self._frames_tracked / self.confidence_cap_frames)

        speed_px_per_s = math.hypot(vx, vy)
        speed_factor = min(1.0, speed_px_per_s / self.speed_full_confidence_px_per_s)

        traj_pts = [(p[1], p[2]) for p in self._buf]
        straightness = _trajectory_straightness(traj_pts)

        confidence = age_factor * speed_factor * straightness
        confidence = float(min(1.0, max(0.0, confidence)))

        return TrackedTarget(
            pixel=(xn, yn),
            velocity=(vx, vy),
            aim_pixel=(ax, ay),
            confidence=confidence,
            age_frames=self._frames_tracked,
        )
