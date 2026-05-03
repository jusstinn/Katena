"""Lock-stability filter that prevents 'aim teleports'.

The fused detection stream can occasionally jump from one object to
another between frames — a YOLO box that snaps to a new drone-like
blob, a motion-detector track that swaps to a closer fast-mover,
or HoverLockDetector handing off between CSRT and motion. If the
laser tracking pipeline naively follows that jump:

  * the predictor estimates a huge spurious velocity (jump_px / dt);
  * the aim diamond teleports across the screen;
  * any active FIRE state would spray the laser through the empty
    space between the old and new target.

`LockFilter` sits between the fused detection and the predictor and
solves this with a simple two-phase rule:

  Phase 1 (steady):
    Each new detection within `max_jump_px` of the previously-accepted
    centroid is accepted normally.

  Phase 2 (suspicion):
    A detection too far away is NOT immediately accepted. It enters a
    "candidate" zone. If subsequent detections stay clustered around
    that candidate for `candidate_frames` consecutive observations, we
    accept it as a confirmed target swap and tell the caller (via
    `discontinuity=True`) so they can reset the predictor and force
    FIRE back to TRACKING. During the suspicion window we return
    `detection=None` so the rest of the pipeline behaves as if it lost
    the target — the orange aim diamond fades, the laser disengages,
    and we re-arm cleanly on the new target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilterResult:
    detection: Any            # Detection to feed downstream, or None if suspicion period
    discontinuity: bool       # True iff we just confirmed a target swap (caller resets state)


class LockFilter:
    def __init__(
        self,
        max_jump_px: float = 250.0,
        candidate_frames: int = 5,
    ) -> None:
        if max_jump_px <= 0:
            raise ValueError("max_jump_px must be > 0")
        if candidate_frames < 1:
            raise ValueError("candidate_frames must be >= 1")
        self.max_jump_px = float(max_jump_px)
        self.candidate_frames = int(candidate_frames)
        self._last_centroid: tuple[float, float] | None = None
        self._cand: tuple[float, float] | None = None
        self._cand_count: int = 0

    @property
    def in_suspicion(self) -> bool:
        return self._cand is not None

    def update(self, det) -> FilterResult:
        if det is None:
            # No detection at all — drop suspicion state. Don't clear
            # last_centroid: when a detection comes back in roughly the
            # same area we still want to treat it as the same target.
            self._cand = None
            self._cand_count = 0
            return FilterResult(detection=None, discontinuity=False)

        cx = float(det.centroid[0])
        cy = float(det.centroid[1])

        if self._last_centroid is None:
            self._last_centroid = (cx, cy)
            return FilterResult(detection=det, discontinuity=False)

        d = math.hypot(cx - self._last_centroid[0], cy - self._last_centroid[1])
        if d <= self.max_jump_px:
            # Normal smooth motion — accept and clear any candidate.
            self._last_centroid = (cx, cy)
            self._cand = None
            self._cand_count = 0
            return FilterResult(detection=det, discontinuity=False)

        # Big jump → enter / continue suspicion period.
        if self._cand is None:
            self._cand = (cx, cy)
            self._cand_count = 1
        else:
            cand_dist = math.hypot(cx - self._cand[0], cy - self._cand[1])
            if cand_dist <= self.max_jump_px:
                self._cand_count += 1
                self._cand = (cx, cy)
            else:
                # Candidate is itself wandering; restart the count
                # at this newest position.
                self._cand = (cx, cy)
                self._cand_count = 1

        if self._cand_count >= self.candidate_frames:
            # Stable enough — accept the swap and emit discontinuity.
            self._last_centroid = (cx, cy)
            self._cand = None
            self._cand_count = 0
            return FilterResult(detection=det, discontinuity=True)

        # Still verifying — suppress this frame's detection.
        return FilterResult(detection=None, discontinuity=False)

    def reset(self) -> None:
        self._last_centroid = None
        self._cand = None
        self._cand_count = 0
