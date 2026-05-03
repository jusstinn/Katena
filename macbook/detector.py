"""Drone detection — OpenCV motion + YOLOv8 ensemble.

Two parallel branches running on every frame:

  Motion branch (always on, primary, reliable):
      MOG2 background subtractor -> morphology -> largest contour
      Detects ANY moving object large enough. Robust against poor
      lighting and works without any training.

  YOLO branch (optional, cosmetic + ensemble):
      yolov8n.pt (or fine-tuned weights). Pretrained COCO has no
      "drone" class so out of the box it sees person/bottle/etc.
      We accept any YOLO detection that overlaps the motion bbox to
      boost confidence. After fine-tuning on the kids' drone, this
      branch becomes the primary classifier.

The detector returns at most one Detection per frame: the best
candidate by an ensemble score. The aim_point property gives the
servo target — the brief's "where the fiber exits under the motors,"
i.e. bottom-center of bbox offset slightly below.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def _straightness(points: list[tuple[float, float]]) -> float:
    """Net displacement / total path length over a polyline.

    1.0 = perfectly linear motion (drone flying through space).
    Near 0 = jittering in place (head turning, hand vibrating, MOG2 noise
    blob whose contour wobbles around a fixed centre). The intuition the
    user pushed: the drone's whole bbox TRANSLATES, while a head-turn
    just changes which pixels MOG2 lights up around a stationary face.
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


@dataclass
class Detection:
    """A single drone detection with everything the tracker needs."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    centroid: tuple[int, int]
    confidence: float
    source: str  # "motion", "yolo", or "ensemble"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def aim_point(self, drop_fraction: float = 0.15) -> tuple[int, int]:
        """Where the servos should aim — fiber exit zone under the motors.

        Bottom-center of bbox, offset DOWN by `drop_fraction` of the
        bbox height. Default 15 %: high enough that the laser dot lands
        in the fiber-spool zone of a typical FPV drone, low enough that
        it won't aim above the airframe at small bbox sizes.
        """
        x1, _, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        ay = y2 + int(drop_fraction * self.height)
        return cx, ay


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class MotionDetector:
    """Background-subtraction motion detector. Picks the largest moving blob."""

    def __init__(
        self,
        min_area: int = 400,
        max_area: int = 200_000,
        var_threshold: int = 25,
        history: int = 200,
        learning_rate: float = -1.0,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> Detection | None:
        mask = self._bg.apply(frame, learningRate=self.learning_rate)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            if area > best_area:
                x, y, w, h = cv2.boundingRect(c)
                best = (x, y, x + w, y + h)
                best_area = area
        if best is None:
            return None

        x1, y1, x2, y2 = best
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        confidence = min(1.0, best_area / 10_000.0)
        return Detection(
            bbox=best, centroid=(cx, cy), confidence=confidence, source="motion"
        )


@dataclass
class _BlobTrack:
    """Internal track for DroneMotionDetector. One per persistent moving blob.

    `smoothed_speed_px_per_frame` is the EWMA of frame-to-frame centroid
    displacement. It's the strongest discriminator we have between drones
    (fast) and faces / hands / curtains (slow). Score uses it directly.
    """
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: int
    aspect_ratio: float
    solidity: float
    age: int                          # total frames since this track was born
    hits: int                         # frames where it was matched to a candidate
    misses: int                       # consecutive frames it was NOT matched
    last_frame: int                   # frame_idx when last matched
    smoothed_speed_px_per_frame: float = 0.0
    last_centroid: tuple[int, int] | None = None  # for inst-velocity calc
    # Last N centroids — used to compute trajectory straightness (drones
    # move linearly; faces turning produce a jittering centroid that
    # never goes anywhere on net).
    centroid_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=12)
    )

    @property
    def straightness(self) -> float:
        return _straightness(list(self.centroid_history))


class DroneMotionDetector:
    """Drone-tuned motion detector. Picks the FASTEST persistent moving blob.

    The vanilla `MotionDetector` returns the LARGEST moving blob on the
    frame, every frame. In a scene where a drone and a person are both
    moving, the person's face/torso is bigger and wins -- so the box
    sticks on the person, not the drone.

    Key insight: **drones move fast**, faces/hands/curtains move slow.
    This detector scores tracks by smoothed pixel-space speed, with
    persistence as a tie-breaker. A drone moving 200 px/s wins against a
    face moving 10 px/s even if the face has been tracked for 10x as
    many frames.

    Pipeline per frame:

    1. MOG2 background subtraction -> binary mask -> morph open + dilate
       to get clean motion blobs.

    2. Mild geometric gates: drop obvious garbage (huge walls, hairline
       contour fragments, very thin shapes). Defaults are PERMISSIVE --
       speed scoring does the real work, so we don't over-filter.

    3. Greedy nearest-centroid matching of candidates against existing
       tracks. New tracks for unmatched candidates. Drop tracks after
       `max_misses` consecutive misses.

    4. Each match updates `smoothed_speed_px_per_frame` (EWMA of
       frame-to-frame centroid jump).

    5. Pick the highest-scoring eligible track:
           score = (smoothed_speed, hits, -area)
       i.e. fast-mover wins; persistence breaks ties; smaller wins last.

    6. **Sticky preferred track.** Once a track is picked, we KEEP it as
       the chosen one as long as it remains eligible -- even if another
       track briefly outscores it. That kills the "box jumps to your
       face for one frame" failure mode when the drone briefly slows.

    All knobs are constructor params so we can tune live from CLI.
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 20_000,     # rejects close-up hands/faces (huge in pixel area)
        min_aspect: float = 0.25,
        max_aspect: float = 4.0,
        min_solidity: float = 0.40,
        var_threshold: int = 30,
        history: int = 200,
        learning_rate: float = 0.005,  # MOG2 auto-ish; faster acquisition
        match_radius_px: int = 120, # generous: a fast drone jumps a lot
        min_hits: int = 2,          # 1-frame latency to filter pure flickers
        max_misses: int = 10,       # forgiving: keep lock through brief occlusion
        speed_alpha: float = 0.4,   # EWMA smoothing on per-track speed
        min_speed_px_per_frame: float = 12.0,
        # Min smoothed speed required to PROMOTE a track to the chosen one.
        # Suppresses faces, finger-covering-the-lens, slow hand waves, etc.
        demote_speed_px_per_frame: float = 3.0,
        # If a sticky/preferred track's smoothed speed drops below THIS, we
        # release the lock (hysteresis: 12 to grab, 3 to release). A
        # hovering drone with tiny micro-movements stays locked; a hand
        # that stops waving releases in ~5 frames.
        sticky_tracks: bool = True, # don't switch off the chosen track unless it dies
        min_straightness: float = 0.45,
        # Trajectory linearity required to PROMOTE a track. Drones move
        # linearly (≈0.8-1.0). A face/head turning makes the contour's
        # centroid wander around a fixed point (≈0.0-0.2). 0.45 catches
        # mild drone curves while killing wander-in-place artefacts.
        # Bypassed for sticky tracks (acrobatic drones may dip under it
        # mid-flight; we don't want to lose lock on real targets).
        prefer_top_half: bool = False,
        # `prefer_smaller` is kept for backward compat but is now only the
        # third-tier tie-breaker; speed dominates.
        prefer_smaller: bool = True,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_solidity = min_solidity
        self.learning_rate = learning_rate
        self.match_radius_px = match_radius_px
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.prefer_top_half = prefer_top_half
        self.prefer_smaller = prefer_smaller
        self.speed_alpha = float(speed_alpha)
        self.min_speed_px_per_frame = float(min_speed_px_per_frame)
        self.demote_speed_px_per_frame = float(demote_speed_px_per_frame)
        self.sticky_tracks = bool(sticky_tracks)
        self.min_straightness = float(min_straightness)

        self._tracks: list[_BlobTrack] = []
        self._next_id = 1
        self._frame_idx = 0
        self._preferred_track_id: int | None = None

    @property
    def active_tracks(self) -> list[_BlobTrack]:
        """Tracks that satisfied min_hits and are currently matched."""
        return [t for t in self._tracks if t.hits >= self.min_hits and t.misses == 0]

    @property
    def all_tracks(self) -> list[_BlobTrack]:
        return list(self._tracks)

    def _candidates(self, frame: np.ndarray) -> list[dict]:
        mask = self._bg.apply(frame, learningRate=self.learning_rate)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[dict] = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            ar = w / h
            if not (self.min_aspect <= ar <= self.max_aspect):
                continue
            hull = cv2.convexHull(c)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            sol = area / hull_area
            if sol < self.min_solidity:
                continue
            out.append({
                "bbox": (x, y, x + w, y + h),
                "centroid": (x + w // 2, y + h // 2),
                "area": area,
                "aspect_ratio": ar,
                "solidity": float(sol),
            })
        return out

    def detect(self, frame: np.ndarray) -> Detection | None:
        self._frame_idx += 1
        candidates = self._candidates(frame)

        # Greedy nearest-centroid matching of tracks <- candidates.
        used: set[int] = set()
        for t in self._tracks:
            best_i, best_d2 = -1, self.match_radius_px * self.match_radius_px + 1
            for i, c in enumerate(candidates):
                if i in used:
                    continue
                dx = c["centroid"][0] - t.centroid[0]
                dy = c["centroid"][1] - t.centroid[1]
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                c = candidates[best_i]
                # Update smoothed speed (EWMA over per-frame centroid jump).
                dx = c["centroid"][0] - t.centroid[0]
                dy = c["centroid"][1] - t.centroid[1]
                inst_speed = (dx * dx + dy * dy) ** 0.5
                a = self.speed_alpha
                t.smoothed_speed_px_per_frame = (
                    (1 - a) * t.smoothed_speed_px_per_frame + a * inst_speed
                )
                t.last_centroid = t.centroid
                t.bbox = c["bbox"]
                t.centroid = c["centroid"]
                t.area = c["area"]
                t.aspect_ratio = c["aspect_ratio"]
                t.solidity = c["solidity"]
                t.hits += 1
                t.misses = 0
                t.last_frame = self._frame_idx
                t.centroid_history.append(
                    (float(c["centroid"][0]), float(c["centroid"][1]))
                )
            else:
                t.misses += 1
            t.age += 1

        # Birth tracks for unmatched candidates.
        for i, c in enumerate(candidates):
            if i in used:
                continue
            new_track = _BlobTrack(
                track_id=self._next_id,
                bbox=c["bbox"], centroid=c["centroid"], area=c["area"],
                aspect_ratio=c["aspect_ratio"], solidity=c["solidity"],
                age=1, hits=1, misses=0, last_frame=self._frame_idx,
                smoothed_speed_px_per_frame=0.0, last_centroid=None,
            )
            new_track.centroid_history.append(
                (float(c["centroid"][0]), float(c["centroid"][1]))
            )
            self._tracks.append(new_track)
            self._next_id += 1

        # Evict dead tracks. If our preferred track is gone, drop the lock.
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        if self._preferred_track_id is not None:
            if not any(t.track_id == self._preferred_track_id for t in self._tracks):
                self._preferred_track_id = None

        eligible = self.active_tracks
        if not eligible:
            return None

        def _score(t: _BlobTrack) -> tuple:
            top_bias = 0
            if self.prefer_top_half:
                top_bias = 1 if t.centroid[1] < frame.shape[0] // 2 else 0
            area_term = -t.area if self.prefer_smaller else t.area
            # SPEED is the primary discriminator. Round to integer px/frame
            # so 19.7 px/frame ties with 20.1 -> persistence breaks the tie.
            speed_term = int(round(t.smoothed_speed_px_per_frame))
            return (top_bias, speed_term, t.hits, area_term)

        # Sticky preference: keep the chosen track as long as it stays
        # eligible AND moving above the demote threshold. Hysteresis
        # between min_speed (promote) and demote_speed prevents flicker.
        chosen = None
        if self.sticky_tracks and self._preferred_track_id is not None:
            for t in eligible:
                if t.track_id == self._preferred_track_id:
                    if t.smoothed_speed_px_per_frame >= self.demote_speed_px_per_frame:
                        chosen = t
                    else:
                        # Track stopped moving meaningfully -> release lock.
                        self._preferred_track_id = None
                    break

        # Fresh selection: must beat the higher promote gate AND look like
        # it's actually translating through space (not jittering in place).
        if chosen is None:
            promotable = [
                t for t in eligible
                if t.smoothed_speed_px_per_frame >= self.min_speed_px_per_frame
                and t.straightness >= self.min_straightness
            ]
            if not promotable:
                # Nothing is moving fast AND straight. Stay silent.
                return None
            chosen = max(promotable, key=_score)
            self._preferred_track_id = chosen.track_id

        # Confidence reflects how "drone-like" the chosen track is:
        # fast + persistent + non-trivial size + LINEAR motion.
        speed_factor = min(1.0, chosen.smoothed_speed_px_per_frame / 25.0)
        hit_factor = min(1.0, chosen.hits / 10.0)
        size_factor = min(1.0, chosen.area / 6_000.0)
        straight_factor = chosen.straightness  # already in [0, 1]
        conf = (
            0.10
            + 0.35 * speed_factor
            + 0.15 * hit_factor
            + 0.10 * size_factor
            + 0.30 * straight_factor
        )
        conf = float(min(1.0, conf))
        return Detection(
            bbox=chosen.bbox,
            centroid=chosen.centroid,
            confidence=conf,
            source="motion",
        )


class YoloDetector:
    """YOLOv8 wrapper. Returns the highest-confidence detection on the frame."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        target_classes: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.device = device

    def detect(self, frame: np.ndarray) -> Detection | None:
        kwargs: dict[str, Any] = {"verbose": False, "conf": self.conf_threshold}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = -1
        best_conf = 0.0
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            if self.target_classes and cls_name not in self.target_classes:
                continue
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        if best_idx < 0:
            return None

        box = boxes[best_idx]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        return Detection(
            bbox=(x1, y1, x2, y2),
            centroid=((x1 + x2) // 2, (y1 + y2) // 2),
            confidence=best_conf,
            source="yolo",
        )


class EnsembleDetector:
    """Runs motion + (optional) YOLO. Combines results sensibly.

    Decision logic:
      - If both fire and bboxes overlap (IoU > 0.2) → ensemble (high confidence)
      - If only YOLO fires above its threshold → use YOLO
      - If only motion fires → use motion
      - If both fire but disagree → prefer motion (more reliable on small drones)
    """

    def __init__(
        self,
        motion: MotionDetector | None = None,
        yolo: YoloDetector | None = None,
        iou_match: float = 0.2,
    ) -> None:
        self.motion = motion or MotionDetector()
        self.yolo = yolo
        self.iou_match = iou_match

    def detect(self, frame: np.ndarray) -> Detection | None:
        motion_det = self.motion.detect(frame)
        yolo_det = self.yolo.detect(frame) if self.yolo else None

        # When YOLO and motion agree we anchor the bbox (and therefore
        # the aim point) to YOLO -- it's tight to the airframe while
        # MOG2's blob wobbles with shadows / lighting / partial
        # occlusion. The boosted confidence still encodes "two
        # independent detectors saw the same object".
        if motion_det and yolo_det:
            iou = _bbox_iou(motion_det.bbox, yolo_det.bbox)
            if iou >= self.iou_match:
                conf = min(1.0, (motion_det.confidence + yolo_det.confidence) / 2.0 + 0.2)
                return Detection(
                    bbox=yolo_det.bbox,
                    centroid=yolo_det.centroid,
                    confidence=conf,
                    source="ensemble",
                )
            return yolo_det
        if yolo_det:
            return yolo_det
        return motion_det


class _TemplateTracker:
    """Tiny appearance tracker using cv2.matchTemplate.

    Used as a fallback when cv2 wasn't built with the contrib trackers
    (CSRT/KCF/etc) — JetPack's stock OpenCV doesn't ship them.

    Stores the bbox patch as a template; on each update searches a
    window around the last bbox and finds the best match. Good enough
    for short hover gaps (a few seconds) where the drone's appearance
    barely changes. Fails gracefully if correlation drops below
    `fail_threshold` (0..1, higher = stricter).
    """

    def __init__(self, search_margin: int = 60, fail_threshold: float = 0.40) -> None:
        self.search_margin = int(search_margin)
        self.fail_threshold = float(fail_threshold)
        self.template: np.ndarray | None = None
        self._wh: tuple[int, int] | None = None
        self._xy: tuple[int, int] | None = None  # top-left

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = (int(v) for v in bbox)
        if w < 6 or h < 6:
            return False
        H, W = frame.shape[:2]
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y, h))
        patch = frame[y:y + h, x:x + w]
        if patch.size == 0:
            return False
        self.template = patch.copy()
        self._xy = (x, y)
        self._wh = (w, h)
        return True

    def update(self, frame: np.ndarray):
        if self.template is None or self._xy is None or self._wh is None:
            return False, None
        x, y = self._xy
        w, h = self._wh
        m = self.search_margin
        H, W = frame.shape[:2]
        sx1 = max(0, x - m)
        sy1 = max(0, y - m)
        sx2 = min(W, x + w + m)
        sy2 = min(H, y + h + m)
        if sx2 - sx1 < w + 2 or sy2 - sy1 < h + 2:
            return False, None
        roi = frame[sy1:sy2, sx1:sx2]
        try:
            res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return False, None
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < self.fail_threshold:
            return False, None
        new_x = sx1 + int(max_loc[0])
        new_y = sy1 + int(max_loc[1])
        self._xy = (new_x, new_y)
        return True, (new_x, new_y, w, h)


class HoverLockDetector:
    """Wraps a base Detector to keep the lock alive when motion stops.

    Use case: a hovering / stationary drone gradually gets absorbed
    into MOG2's background model and stops triggering motion. We don't
    want the box to disappear in that case — we want to keep aiming at
    the drone.

    Strategy: maintain an OpenCV CSRT tracker (template-correlation
    based, robust to small drift). On every frame:

      1. Run the base detector. If it fires, that's authoritative —
         re-seed the tracker with the new bbox and emit it.

      2. If base is silent BUT we have a recent lock, advance the
         tracker on the new frame and emit its bbox (lower confidence
         to signal "this is hover-tracking, not detection").

      3. If the tracker fails or we've been hovering longer than
         `max_hover_s`, give up — emit None and wait for motion to
         re-acquire from scratch.

    A tracker is only seeded when the base detection's `confidence`
    >= `seed_confidence` so we don't lock onto noise.

    On the Jetson, CSRT runs ~3-8 ms per frame at 720p which is fine
    when it's only used during hover gaps. Falls back gracefully if
    cv2 was built without it.
    """

    def __init__(
        self,
        base: Any,
        max_hover_s: float = 4.0,
        seed_confidence: float = 0.45,
        hover_confidence: float = 0.55,
        clock=None,
    ) -> None:
        import time
        self.base = base
        self.max_hover_s = float(max_hover_s)
        self.seed_confidence = float(seed_confidence)
        self.hover_confidence = float(hover_confidence)
        self._clock = clock or time.perf_counter
        self._tracker = None
        self._last_motion_at = 0.0
        self._last_bbox: tuple[int, int, int, int] | None = None
        # Counter of consecutive hover-only emits, exposed for diagnostics.
        self.hover_streak: int = 0

    @property
    def is_hovering(self) -> bool:
        return self._tracker is not None

    @staticmethod
    def _make_tracker():
        # Prefer CSRT if available (opencv-contrib). Most JetPack /
        # vanilla opencv-python builds DON'T ship contrib trackers, so
        # we fall back to a small matchTemplate-based tracker which
        # works on stock cv2 and is plenty for short hover gaps.
        try:
            return cv2.legacy.TrackerCSRT_create()
        except (AttributeError, cv2.error):
            pass
        try:
            return cv2.TrackerCSRT_create()
        except (AttributeError, cv2.error):
            pass
        return _TemplateTracker()

    def _seed(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        tracker = self._make_tracker()
        if tracker is None:
            self._tracker = None
            return
        try:
            tracker.init(frame, (int(x1), int(y1), int(w), int(h)))
            self._tracker = tracker
            self._last_bbox = bbox
        except cv2.error:
            self._tracker = None

    def detect(self, frame: np.ndarray) -> Detection | None:
        det = self.base.detect(frame)
        now = self._clock()
        if det is not None:
            # Live detection wins. Re-seed tracker so it always reflects
            # the latest known bbox/appearance.
            self._last_motion_at = now
            self.hover_streak = 0
            if det.confidence >= self.seed_confidence:
                self._seed(frame, det.bbox)
            self._last_bbox = det.bbox
            return det

        # Base went silent. Try the hover tracker.
        if self._tracker is None or self._last_bbox is None:
            return None
        if (now - self._last_motion_at) > self.max_hover_s:
            self._tracker = None
            self._last_bbox = None
            self.hover_streak = 0
            return None
        try:
            ok, bbox = self._tracker.update(frame)
        except cv2.error:
            ok = False
            bbox = None
        if not ok or bbox is None:
            self._tracker = None
            self._last_bbox = None
            self.hover_streak = 0
            return None
        x, y, w, h = (int(v) for v in bbox)
        if w <= 1 or h <= 1:
            self._tracker = None
            return None
        self._last_bbox = (x, y, x + w, y + h)
        self.hover_streak += 1
        return Detection(
            bbox=self._last_bbox,
            centroid=(x + w // 2, y + h // 2),
            confidence=self.hover_confidence,
            source="hover",
        )
