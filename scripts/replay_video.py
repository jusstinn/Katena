"""Replay a recorded video through the Katena detector stack.

Pipes any video file (drone-flight footage, archived camera capture,
whatever) through the SAME `EnsembleDetector` (motion + YOLO) that
`main_tracker.py` uses live. Lets you see, frame by frame, what the
production pipeline would think about your footage:

  - What does base YOLOv8 call your drone? (probably "airplane", "kite",
    "bird" — that's how you know how much fine-tuning you need)
  - Does the motion branch reliably catch it?
  - Where does the AIM crosshair land relative to the actual fiber-exit
    zone?

Three useful side outputs:

  --save              Write an annotated MP4 next to the source. Use
                      this clip directly in your pitch slides.
  --export-labels     For every frame with a confident detection, write
                      a YOLO-format .txt label (class 0 = drone) to
                      <video>_labels/. Drop those + the frames into a
                      training run and you've auto-bootstrapped your
                      dataset.
  --all-yolo          Show ALL YOLO classes above the threshold (not
                      just the top one). Useful for seeing what other
                      classes the model is confused with.

Keys (live preview):
    Space         pause / resume
    Right / .     step one frame forward (works while paused)
    Left  / ,     step one frame backward
    R             jump back to start
    S             save current annotated frame to recordings/
    H             toggle help overlay
    Q / Esc       quit

Usage:
    python scripts/replay_video.py recordings/drone_flight.mp4
    python scripts/replay_video.py recordings/drone_flight.mp4 --save
    python scripts/replay_video.py recordings/drone_flight.mp4 --export-labels
    python scripts/replay_video.py recordings/drone_flight.mp4 --weights runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from macbook.detector import (  # noqa: E402
    Detection,
    DroneMotionDetector,
    MotionDetector,
    YoloDetector,
    _bbox_iou,
)
from macbook.overlay import _shadowed_text  # noqa: E402

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN = (220, 220, 0)
_YELLOW = (0, 220, 220)
_GREEN = (0, 220, 0)
_RED = (0, 0, 220)
_PURPLE = (200, 100, 255)


def _fmt_pct(x: float) -> str:
    return f"{int(round(x * 100)):3d}%"


def _fuse(motion_det: Detection | None, yolo_det: Detection | None) -> Detection | None:
    """Same fusion rules as `EnsembleDetector.detect`, factored out so we can
    reuse the underlying motion/yolo objects for the side-panel display.

    YOLO's bbox is tighter on the airframe than MOG2's wobbling motion
    blob, so when YOLO concurs we anchor the AIM POINT to YOLO's bbox.
    Priority: ensemble (YOLO bbox + boosted conf) > YOLO-only > motion.
    """
    if motion_det and yolo_det:
        if _bbox_iou(motion_det.bbox, yolo_det.bbox) >= 0.2:
            return Detection(
                bbox=yolo_det.bbox,
                centroid=yolo_det.centroid,
                confidence=min(1.0, (motion_det.confidence + yolo_det.confidence) / 2.0 + 0.2),
                source="ensemble",
            )
        return yolo_det
    if yolo_det:
        return yolo_det
    return motion_det


def _draw_motion_box(frame: np.ndarray, det: Detection | None) -> None:
    if det is None or det.source != "motion":
        return
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), _CYAN, 1, cv2.LINE_AA)
    _shadowed_text(
        frame,
        f"motion {_fmt_pct(det.confidence)} area={det.area}",
        (x1, max(15, y1 - 6)),
        scale=0.45,
        color=_CYAN,
        thickness=1,
    )


def _draw_yolo_boxes(
    frame: np.ndarray,
    boxes,
    names: dict[int, str],
    conf_threshold: float,
    show_all: bool,
) -> list[tuple[str, float]]:
    """Draw YOLO boxes; return list of (class_name, conf) for the side panel."""
    out: list[tuple[str, float]] = []
    if boxes is None or len(boxes) == 0:
        return out
    if not show_all:
        # Only draw the single highest-confidence box (matches production)
        best_i = max(range(len(boxes)), key=lambda i: float(boxes[i].conf[0]))
        boxes_iter = [boxes[best_i]]
    else:
        boxes_iter = list(boxes)

    for box in boxes_iter:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        name = names.get(cls_id, str(cls_id))
        out.append((name, conf))
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = _GREEN if conf >= 0.6 else (_YELLOW if conf >= 0.4 else _PURPLE)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _shadowed_text(
            frame,
            f"{name} {_fmt_pct(conf)}",
            (x1, max(15, y1 - 6)),
            scale=0.5,
            color=color,
            thickness=1,
        )
    return out


def _draw_fire_overlay(frame, zone, trail, laser_xy, *, max_age_s: float = 1.2) -> None:
    from macbook.sweep import draw_fire_overlay
    draw_fire_overlay(frame, zone, trail, laser_xy, trail_max_age_s=max_age_s)


def _draw_fire_status(frame, decision) -> None:
    from macbook.sweep import draw_fire_status
    draw_fire_status(frame, decision, text_fn=_shadowed_text)


def _draw_aim(frame: np.ndarray, det: Detection | None) -> None:
    if det is None:
        return
    ax, ay = det.aim_point()
    ay = min(ay, frame.shape[0] - 5)
    cv2.drawMarker(frame, (ax, ay), _RED, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 12, _RED, 1, cv2.LINE_AA)
    _shadowed_text(frame, "AIM", (ax + 16, ay + 4), scale=0.45, color=_RED, thickness=1)


def _draw_side_panel(
    frame: np.ndarray,
    yolo_dets: list[tuple[str, float]],
    motion_det: Detection | None,
    fused: Detection | None,
    frame_idx: int,
    total: int,
    fps_actual: float,
    yolo_ms: float,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 290
    x0 = w - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 10, y0 + 220), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + 220), _WHITE, 1)

    y = y0 + 22
    _shadowed_text(frame, "DETECTOR INSPECTOR", (x0 + 12, y), scale=0.55, color=_WHITE, thickness=1)
    y += 22
    _shadowed_text(
        frame,
        f"frame {frame_idx + 1}/{total}   {fps_actual:4.1f} fps",
        (x0 + 12, y),
        scale=0.45,
        color=(180, 180, 180),
        thickness=1,
    )
    y += 22

    if motion_det is not None:
        _shadowed_text(frame, f"MOTION  {_fmt_pct(motion_det.confidence)}  a={motion_det.area}",
                       (x0 + 12, y), scale=0.5, color=_CYAN, thickness=1)
    else:
        _shadowed_text(frame, "MOTION  —", (x0 + 12, y), scale=0.5, color=(120, 120, 120), thickness=1)
    y += 22

    _shadowed_text(frame, f"YOLO  ({yolo_ms:.0f} ms)", (x0 + 12, y), scale=0.5, color=_GREEN, thickness=1)
    y += 18
    if not yolo_dets:
        _shadowed_text(frame, "  no detections", (x0 + 12, y),
                       scale=0.45, color=(120, 120, 120), thickness=1)
        y += 18
    else:
        for name, conf in yolo_dets[:6]:
            _shadowed_text(frame, f"  {name:<14s} {_fmt_pct(conf)}",
                           (x0 + 12, y), scale=0.45, color=_WHITE, thickness=1)
            y += 18

    y += 4
    if fused is not None:
        col = _RED if fused.source == "ensemble" else _YELLOW
        _shadowed_text(frame, f"FUSED ({fused.source}) {_fmt_pct(fused.confidence)}",
                       (x0 + 12, y), scale=0.5, color=col, thickness=1)
    else:
        _shadowed_text(frame, "FUSED  —", (x0 + 12, y), scale=0.5, color=(120, 120, 120), thickness=1)


def _write_label_file(
    out_dir: Path,
    frame_idx: int,
    det: Detection | None,
    frame_w: int,
    frame_h: int,
) -> bool:
    """Write a YOLO-format label file. class 0 = drone. Returns True if written."""
    if det is None:
        return False
    x1, y1, x2, y2 = det.bbox
    bx = (x1 + x2) / 2.0 / frame_w
    by = (y1 + y2) / 2.0 / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h
    if bw <= 0 or bh <= 0:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / f"frame_{frame_idx:06d}.txt"
    label_path.write_text(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt",
                        help="YOLO weights path (default: project root yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--motion-min-area", type=int, default=400)
    parser.add_argument("--no-yolo", action="store_true", help="Run motion-only")
    parser.add_argument("--drone-motion", action="store_true",
                        help="Use the drone-tuned motion detector "
                             "(geometric filters + persistence tracking) instead "
                             "of the legacy biggest-blob detector.")
    parser.add_argument("--motion-max-area", type=int, default=60_000,
                        help="DroneMotionDetector: max blob area (default 60k = very "
                             "permissive; speed scoring does the real filtering)")
    parser.add_argument("--motion-min-hits", type=int, default=2,
                        help="DroneMotionDetector: min matches to emit (default 2)")
    parser.add_argument("--motion-max-misses", type=int, default=10,
                        help="DroneMotionDetector: drop a track after this many missed frames")
    parser.add_argument("--motion-min-aspect", type=float, default=0.25)
    parser.add_argument("--motion-max-aspect", type=float, default=4.0)
    parser.add_argument("--motion-min-solidity", type=float, default=0.40)
    parser.add_argument("--motion-learning-rate", type=float, default=0.005)
    parser.add_argument("--motion-min-speed", type=float, default=12.0,
                        help="Min smoothed speed (px/frame) to PROMOTE a track. "
                             "~12 silences hands/faces/lens-cover; lower if drone is slow.")
    parser.add_argument("--motion-demote-speed", type=float, default=3.0,
                        help="Speed (px/frame) below which the lock is RELEASED.")
    parser.add_argument("--motion-min-straightness", type=float, default=0.45,
                        help="Trajectory linearity to PROMOTE a track. "
                             "1.0=linear (drone), 0=jitter-in-place (face turn).")
    parser.add_argument("--hover-lock", action="store_true",
                        help="Wrap detector with HoverLockDetector (CSRT) so a "
                             "hovering drone keeps the lock when motion stops.")
    parser.add_argument("--hover-max-s", type=float, default=4.0,
                        help="Max hover seconds without motion before lock drops.")
    parser.add_argument("--motion-no-sticky", action="store_true",
                        help="Disable track stickiness (default = sticky)")
    parser.add_argument("--motion-prefer-top", action="store_true",
                        help="DroneMotionDetector: bias toward upper-half-of-frame tracks")
    parser.add_argument("--motion-prefer-bigger", action="store_true",
                        help="DroneMotionDetector: on equal speed/persistence prefer LARGER blob")
    parser.add_argument("--lead-time", type=float, default=0.0,
                        help="TargetPredictor: seconds to lead the target. 0.0 disables. "
                             "Try 0.10-0.25 once servo latency is measured.")
    parser.add_argument("--cable-offset-x", type=float, default=0.0,
                        help="Aim offset (px) from drone centroid in X")
    parser.add_argument("--cable-offset-y", type=float, default=0.0,
                        help="Aim offset (px) from drone centroid in Y "
                             "(positive = below drone, toward the trailing fiber)")
    # --- Laser sweep / FIRE visualization (mirrors jetson_live_detect) ---
    parser.add_argument("--fire", action="store_true",
                        help="Enable laser sweep planner + FIRE state machine.")
    parser.add_argument("--fire-pattern", default="lissajous",
                        choices=["lissajous", "horizontal", "vertical",
                                 "circle", "figure_eight", "static"],
                        help="Sweep pattern inside the aim zone.")
    parser.add_argument("--fire-zone-w", type=float, default=0.5,
                        help="Aim zone width as fraction of drone bbox width.")
    parser.add_argument("--fire-zone-h", type=float, default=1.0,
                        help="Aim zone height as fraction of drone bbox height.")
    parser.add_argument("--fire-amp-x", type=float, default=0.7,
                        help="Sweep X amplitude as fraction of zone half-width.")
    parser.add_argument("--fire-amp-y", type=float, default=0.7,
                        help="Sweep Y amplitude as fraction of zone half-height.")
    parser.add_argument("--fire-freq-x", type=float, default=1.0,
                        help="Sweep X frequency in Hz (keep <= ~1.5 for real servos).")
    parser.add_argument("--fire-freq-y", type=float, default=0.7,
                        help="Sweep Y frequency in Hz (keep <= ~1.5 for real servos).")
    parser.add_argument("--fire-trail-len", type=int, default=200,
                        help="Deque capacity for the trail. With sub-frame "
                             "interpolation we add ~30 samples/sec regardless "
                             "of producer fps; renderer prunes by max-age.")
    parser.add_argument("--fire-trail-max-age", type=float, default=1.2,
                        help="Seconds of trail kept visible. Lower = more "
                             "compact trail under the drone (default 1.2s).")
    parser.add_argument("--fire-arm-conf", type=float, default=0.45,
                        help="Detection conf needed (sustained) to ARM. Down from "
                             "0.55 because gating uses the fused detection conf now, "
                             "not the predictor's harsh multiplicative score.")
    parser.add_argument("--fire-arm-dwell-s", type=float, default=0.30,
                        help="Sustained-conf seconds before ARM. Down from 0.8s.")
    parser.add_argument("--fire-disarm-conf", type=float, default=0.20)
    parser.add_argument("--fire-override", default="auto",
                        choices=["auto", "force_on", "force_off"],
                        help="Manual override for FIRE state (force_on shows the "
                             "sweep regardless of confidence dwell).")
    parser.add_argument(
        "--lock-filter", action=argparse.BooleanOptionalAction, default=True,
        help="Block sudden centroid teleports between frames so the laser "
             "doesn't chase target swaps. Default: ON. Use --no-lock-filter to disable.",
    )
    parser.add_argument("--lock-max-jump-px", type=float, default=250.0,
                        help="Pixel distance above which a frame-to-frame centroid "
                             "change is treated as a candidate target swap.")
    parser.add_argument("--lock-cand-frames", type=int, default=5,
                        help="Frames the new candidate must remain stable before "
                             "the swap is accepted (and predictor/FIRE reset).")
    parser.add_argument("--all-yolo", action="store_true",
                        help="Show every YOLO box above threshold (not just the top)")
    parser.add_argument("--save", action="store_true",
                        help="Write annotated MP4 next to the source video")
    parser.add_argument("--export-labels", action="store_true",
                        help="Dump YOLO-format labels for confident frames into <video>_labels/")
    parser.add_argument("--export-frames", action="store_true",
                        help="Also dump the matching frames as JPGs into <video>_frames/ "
                             "(use with --export-labels to get a complete training pack)")
    parser.add_argument("--label-conf", type=float, default=0.30,
                        help="Min confidence to count a detection as a training label")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = run to end)")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless: process the file without showing the preview window")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"FAIL: video not found at {args.video}")
        return 1

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"FAIL: OpenCV could not open {args.video}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.video}  {width}x{height} @ {src_fps:.1f} fps  ({total} frames)")

    if args.drone_motion:
        motion = DroneMotionDetector(
            min_area=args.motion_min_area,
            max_area=args.motion_max_area,
            min_aspect=args.motion_min_aspect,
            max_aspect=args.motion_max_aspect,
            min_solidity=args.motion_min_solidity,
            learning_rate=args.motion_learning_rate,
            min_hits=args.motion_min_hits,
            max_misses=args.motion_max_misses,
            min_speed_px_per_frame=args.motion_min_speed,
            demote_speed_px_per_frame=args.motion_demote_speed,
            min_straightness=args.motion_min_straightness,
            sticky_tracks=not args.motion_no_sticky,
            prefer_top_half=args.motion_prefer_top,
            prefer_smaller=not args.motion_prefer_bigger,
        )
        print(f"Using DroneMotionDetector "
              f"(sticky={not args.motion_no_sticky} "
              f"speed grab/release={args.motion_min_speed}/{args.motion_demote_speed} px/f "
              f"min_straight={args.motion_min_straightness} "
              f"area=[{args.motion_min_area},{args.motion_max_area}] "
              f"min_hits={args.motion_min_hits} max_misses={args.motion_max_misses})")
    else:
        motion = MotionDetector(min_area=args.motion_min_area)

    if args.hover_lock:
        from macbook.detector import HoverLockDetector
        motion = HoverLockDetector(motion, max_hover_s=args.hover_max_s)
        print(f"Hover-lock ENABLED (CSRT; max {args.hover_max_s}s)")

    from macbook.tracker import TargetPredictor, cable_aim_offset  # local: keeps cold start fast
    predictor: TargetPredictor | None = None
    if args.lead_time > 0 or args.cable_offset_x != 0 or args.cable_offset_y != 0:
        predictor = TargetPredictor(lead_time_s=max(args.lead_time, 1e-6))
        print(f"Using TargetPredictor lead_time={args.lead_time}s "
              f"cable_offset=({args.cable_offset_x},{args.cable_offset_y})")

    lock_filter = None
    if args.lock_filter:
        from macbook.aim_filter import LockFilter
        lock_filter = LockFilter(
            max_jump_px=args.lock_max_jump_px,
            candidate_frames=args.lock_cand_frames,
        )
        print(f"LockFilter ON  max_jump={args.lock_max_jump_px}px "
              f"candidate_frames={args.lock_cand_frames}")

    sweep_planner = None
    fire_ctl = None
    if args.fire:
        from macbook.sweep import SweepPlanner
        from macbook.fire_state import FireController
        sweep_planner = SweepPlanner(
            pattern=args.fire_pattern,
            width_frac=args.fire_zone_w,
            height_frac=args.fire_zone_h,
            amp_x_frac=args.fire_amp_x,
            amp_y_frac=args.fire_amp_y,
            freq_x_hz=args.fire_freq_x,
            freq_y_hz=args.fire_freq_y,
        )
        fire_ctl = FireController(
            arm_conf=args.fire_arm_conf,
            disarm_conf=args.fire_disarm_conf,
            arm_dwell_s=args.fire_arm_dwell_s,
        )
        print(f"FIRE enabled: pattern={args.fire_pattern} "
              f"zone={args.fire_zone_w}x{args.fire_zone_h} "
              f"freqs={args.fire_freq_x},{args.fire_freq_y}Hz "
              f"override={args.fire_override}")
    yolo_model = None
    yolo_names: dict[int, str] = {}
    if not args.no_yolo:
        if not args.weights.exists():
            print(f"FAIL: YOLO weights not found at {args.weights}")
            return 1
        print(f"Loading YOLO weights from {args.weights}...")
        yolo_inst = YoloDetector(weights=str(args.weights), conf_threshold=args.conf)
        yolo_model = yolo_inst.model
        yolo_names = dict(yolo_model.names)
        print(f"Loaded {len(yolo_names)} classes")

    writer: cv2.VideoWriter | None = None
    save_path: Path | None = None
    if args.save:
        save_path = args.video.with_name(args.video.stem + "_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, src_fps, (width, height))
        print(f"Writing annotated video to {save_path}")

    label_dir = args.video.with_name(args.video.stem + "_labels") if args.export_labels else None
    frame_dir = args.video.with_name(args.video.stem + "_frames") if args.export_frames else None
    if label_dir:
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting labels (>={args.label_conf:.2f} conf) to {label_dir}")
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting frames to {frame_dir}")

    laser_trail: deque[tuple[float, tuple[float, float]]] = deque(
        maxlen=args.fire_trail_len if args.fire else 1
    )

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    window = "Katena Replay  (space=pause, ←/→=step, s=save frame, q=quit)"
    if not args.no_window:
        cv2.namedWindow(window)

    paused = False
    frame_idx = args.start_frame
    last_t = time.perf_counter()
    smoothed_fps = 0.0
    yolo_class_counter: Counter[str] = Counter()
    labels_written = 0
    show_help = True

    def step(direction: int) -> tuple[bool, np.ndarray | None]:
        nonlocal frame_idx
        if direction < 0:
            new_idx = max(0, frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
        ret, fr = cap.read()
        if not ret:
            return False, None
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return True, fr

    try:
        while True:
            if not paused:
                ok, frame = step(1)
                if not ok:
                    print("End of video.")
                    break
                if args.max_frames and (frame_idx - args.start_frame) >= args.max_frames:
                    print(f"Stopped at --max-frames={args.max_frames}")
                    break

            motion_det = motion.detect(frame)

            t_y0 = time.perf_counter()
            yolo_dets: list[tuple[str, float]] = []
            yolo_top: Detection | None = None
            if yolo_model is not None:
                results = yolo_model(frame, verbose=False, conf=args.conf)
                boxes = results[0].boxes if results else None
                yolo_dets = _draw_yolo_boxes(
                    frame, boxes, yolo_names, args.conf, args.all_yolo,
                )
                for name, _ in yolo_dets:
                    yolo_class_counter[name] += 1
                if boxes is not None and len(boxes) > 0:
                    best_i = max(range(len(boxes)), key=lambda i: float(boxes[i].conf[0]))
                    box = boxes[best_i]
                    conf = float(box.conf[0])
                    if conf >= args.conf:
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        yolo_top = Detection(
                            bbox=(x1, y1, x2, y2),
                            centroid=((x1 + x2) // 2, (y1 + y2) // 2),
                            confidence=conf,
                            source="yolo",
                        )
            yolo_ms = (time.perf_counter() - t_y0) * 1000.0

            fused = _fuse(motion_det, yolo_top)

            _draw_motion_box(frame, motion_det)

            # Stability filter: suppresses sudden centroid teleports
            # (target swaps) so the predictor / FIRE don't chase them.
            if lock_filter is not None:
                fr = lock_filter.update(fused)
                if fr.discontinuity:
                    if predictor is not None:
                        predictor.reset()
                    if fire_ctl is not None:
                        fire_ctl.reset()
                    if sweep_planner is not None:
                        sweep_planner.reset()
                    laser_trail.clear()
                fused_for_aim = fr.detection
                if lock_filter.in_suspicion:
                    _shadowed_text(
                        frame, "LOCK SUSPECT", (12, 30),
                        scale=0.55, color=(60, 200, 255), thickness=2,
                    )
            else:
                fused_for_aim = fused

            _draw_aim(frame, fused_for_aim)

            tracked = None
            if predictor is not None:
                if fused_for_aim is not None:
                    cx, cy = fused_for_aim.centroid
                    tracked = predictor.update(time.perf_counter(), float(cx), float(cy))
                    aim_x, aim_y = cable_aim_offset(
                        tracked.aim_pixel,
                        offset_x=args.cable_offset_x,
                        offset_y=args.cable_offset_y,
                    )
                    aim_x_i, aim_y_i = int(aim_x), int(aim_y)
                    if 0 <= aim_x_i < width and 0 <= aim_y_i < height:
                        cv2.drawMarker(frame, (aim_x_i, aim_y_i),
                                       (0, 165, 255), cv2.MARKER_DIAMOND, 28, 2, cv2.LINE_AA)
                        cv2.line(frame, (cx, cy), (aim_x_i, aim_y_i),
                                 (0, 165, 255), 1, cv2.LINE_AA)
                        _shadowed_text(frame,
                                       f"PRED {tracked.speed_px_per_s:5.0f} px/s "
                                       f"conf={tracked.confidence:.2f}",
                                       (aim_x_i + 16, aim_y_i - 6),
                                       scale=0.45,
                                       color=(0, 165, 255),
                                       thickness=1)
                else:
                    if predictor.is_active:
                        predictor.reset()

            if sweep_planner is not None and fire_ctl is not None:
                # Use the detection's own confidence (already composite
                # "drone-likelihood") to gate FIRE, not the predictor's
                # multiplicative score (age * speed * straightness was
                # so harsh that auto-arm almost never triggered for
                # hovering / curving drones).
                conf_for_fire = (
                    fused_for_aim.confidence if fused_for_aim is not None else None
                )
                fire_decision = fire_ctl.update(conf_for_fire, args.fire_override)
                zone = None
                laser_xy = None
                if fused_for_aim is not None and fire_decision.laser_on:
                    velocity = (
                        tracked.velocity if tracked is not None else (0.0, 0.0)
                    )
                    zone = sweep_planner.zone(fused_for_aim.bbox, velocity)
                    now_t = time.perf_counter()
                    # Sub-frame trail interpolation + zone-local trail
                    # storage so the trail stays compact under the
                    # drone instead of streaking across the screen.
                    # Same code path as jetson_live.
                    if laser_trail:
                        last_t = laser_trail[-1][0]
                        dt = now_t - last_t
                        if dt > 0.5:
                            local_xy = sweep_planner.aim_point_local(now_t, zone)
                            laser_trail.append((now_t, local_xy))
                        else:
                            n_sub = max(1, min(10, int(round(dt / 0.033))))
                            for k in range(1, n_sub + 1):
                                t_sub = last_t + (k / n_sub) * dt
                                local_sub = sweep_planner.aim_point_local(t_sub, zone)
                                laser_trail.append((t_sub, local_sub))
                    else:
                        local_xy = sweep_planner.aim_point_local(now_t, zone)
                        laser_trail.append((now_t, local_xy))
                    laser_xy = sweep_planner.local_to_screen(
                        laser_trail[-1][1], zone,
                    )
                # Render even on disengage frames -- the renderer
                # age-fades old segments so the arc disappears
                # gracefully instead of being wiped instantly.
                _draw_fire_overlay(
                    frame, zone, laser_trail, laser_xy,
                    max_age_s=args.fire_trail_max_age,
                )
                _draw_fire_status(frame, fire_decision)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps if smoothed_fps else inst_fps

            _draw_side_panel(frame, yolo_dets, motion_det, fused,
                             frame_idx, total, smoothed_fps, yolo_ms)

            if paused:
                _shadowed_text(frame, "PAUSED", (12, 30), scale=0.7,
                               color=_YELLOW, thickness=2)
            if show_help and not args.no_window:
                _shadowed_text(
                    frame,
                    "space pause   < > step   r restart   s snapshot   h hide   q quit",
                    (12, height - 12),
                    scale=0.45,
                    thickness=1,
                )

            if writer is not None:
                writer.write(frame)

            if label_dir is not None and fused is not None and fused.confidence >= args.label_conf:
                if _write_label_file(label_dir, frame_idx, fused, width, height):
                    labels_written += 1
                    if frame_dir is not None:
                        cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), frame)

            if args.no_window:
                if frame_idx % 30 == 0:
                    print(f"  frame {frame_idx}/{total}  fps={smoothed_fps:.1f}  labels={labels_written}")
                continue

            cv2.imshow(window, frame)
            wait = 0 if paused else max(1, int(1000 / src_fps) - int(yolo_ms))
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("."), 83):  # right arrow varies by build
                paused = True
                step(1)
            elif key in (ord(","), 81):  # left arrow
                paused = True
                step(-1)
            elif key == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
                frame_idx = args.start_frame
                paused = False
            elif key == ord("s"):
                snap = PROJECT_ROOT / "recordings" / f"replay_snap_{frame_idx:06d}.png"
                snap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), frame)
                print(f"Saved {snap}")
            elif key == ord("h"):
                show_help = not show_help

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    print()
    print("=== SUMMARY ===")
    print(f"Frames processed:  {frame_idx - args.start_frame + 1}")
    if yolo_class_counter:
        print("Top YOLO classes seen (across all frames):")
        for name, cnt in yolo_class_counter.most_common(10):
            print(f"  {name:<20s} {cnt}")
    else:
        print("No YOLO detections (or --no-yolo).")
    if writer is not None and save_path is not None:
        print(f"Annotated MP4:     {save_path}")
    if label_dir is not None:
        print(f"Labels written:    {labels_written}  ({label_dir})")
    if frame_dir is not None:
        print(f"Frames written:    {frame_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
