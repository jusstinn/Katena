"""Live camera + YOLO + motion-detection preview on the Jetson.

Same detection stack as `main_tracker.py` (motion MOG2 + YOLOv8 ensemble),
but stripped of the state machine, servo control, and Pico link — pure
"point camera at thing, see what the model thinks" loop. Use this on
the CASK to confirm the camera + detector pipeline works end-to-end
before wiring up the rig.

Two display modes:

  Default       Live OpenCV window with the full annotated overlay.
                Requires a display (an HDMI screen on the Jetson, or
                `ssh -X cask`). Press q to quit, s to snapshot, h for
                help.

  --headless    No window. Writes the latest annotated frame to
                `recordings/jetson_live_latest.jpg` every frame and
                logs a one-line detection summary to stdout every
                second. Run this when you're SSH'd in without X11.

Camera sources:

  Default       USB UVC camera at index 0 (uses V4L2 backend, MJPG
                pixel format — same as scripts/jetson_record_clip.py).

  --csi         CSI ribbon camera via NVIDIA's nvarguscamerasrc
                GStreamer pipeline. Use this for the official Jetson
                camera modules.

  --gst "..."   Pass an arbitrary GStreamer pipeline string. Use this
                for IP cameras, multi-stream rigs, or non-default CSI
                configurations.

  --index N     Use a different USB camera index (default 0).

Performance notes:

  - Without CUDA (cuda=False), YOLO runs on CPU at ~3-8 fps depending
    on input size. Motion detection is unaffected and runs at full fps.
    Lower YOLO load by passing --yolo-every 3 (run YOLO every 3rd
    frame; reuse last result on the others) or --no-yolo.
  - Once the Jetson has CUDA-enabled torch, expect 25-40 fps end-to-end.

Usage:
    python3 scripts/jetson_live_detect.py
    python3 scripts/jetson_live_detect.py --csi
    python3 scripts/jetson_live_detect.py --headless --yolo-every 3
    python3 scripts/jetson_live_detect.py --weights runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
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
_DIM = (120, 120, 120)
_CYAN = (220, 220, 0)
_YELLOW = (0, 220, 220)
_GREEN = (0, 220, 0)
_RED = (0, 0, 220)
_PURPLE = (200, 100, 255)


def _build_csi_pipeline(width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.gst:
        print(f"Camera: GStreamer pipeline (custom)")
        cap = cv2.VideoCapture(args.gst, cv2.CAP_GSTREAMER)
    elif args.csi:
        pipeline = _build_csi_pipeline(args.width, args.height, args.fps)
        print(f"Camera: CSI via GStreamer")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # Pick the right OS backend: V4L2 on Linux (Jetson), AVFoundation
        # on macOS (so this script can be smoke-tested on a MacBook webcam),
        # CAP_ANY everywhere else.
        import sys
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
            backend_name = "AVFoundation (macOS)"
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2
            backend_name = "V4L2"
        else:
            backend = cv2.CAP_ANY
            backend_name = "CAP_ANY"
        print(f"Camera: USB index {args.index} ({backend_name})")
        cap = cv2.VideoCapture(args.index, backend)
        if backend == cv2.CAP_V4L2:
            # MJPG fourcc is the magic that lets a UVC webcam hit 30 fps at
            # 720p+ on Jetson; not supported (or needed) on macOS.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def _fmt_pct(x: float) -> str:
    return f"{int(round(x * 100)):3d}%"


def _fuse(motion_det: Detection | None, yolo_det: Detection | None) -> Detection | None:
    """Combine motion + YOLO detections into one aim target.

    YOLO's bounding box is much tighter on the actual airframe than
    MOG2's blob (which wobbles with shadows / lighting / partial
    occlusion). So whenever YOLO concurs with motion (their boxes
    overlap), the AIM POINT is computed from YOLO's bbox, not motion's
    -- the laser sits cleanly under the drone instead of drifting with
    the motion blob's edge.

    Priority order:
      1. Both fire AND overlap (IoU >= 0.2)  -> ensemble: YOLO bbox,
         confidence-boosted to encode "two independent detectors agree".
      2. Both fire but DISAGREE (separate objects)  -> prefer YOLO; it's
         the one trained to recognise drones specifically.
      3. Only YOLO  -> YOLO.
      4. Only motion  -> motion (last resort, before YOLO is loaded
         or when the drone is too small for the model).
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
        f"motion {_fmt_pct(det.confidence)} a={det.area}",
        (x1, max(15, y1 - 6)),
        scale=0.45, color=_CYAN, thickness=1,
    )


def _draw_yolo_box(
    frame: np.ndarray,
    det: Detection | None,
    label: str,
) -> None:
    if det is None:
        return
    x1, y1, x2, y2 = det.bbox
    color = _GREEN if det.confidence >= 0.6 else (_YELLOW if det.confidence >= 0.4 else _PURPLE)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    _shadowed_text(
        frame,
        f"{label} {_fmt_pct(det.confidence)}",
        (x1, max(15, y1 - 6)),
        scale=0.5, color=color, thickness=1,
    )


def _draw_aim(frame: np.ndarray, det: Detection | None) -> None:
    if det is None:
        return
    ax, ay = det.aim_point()
    ay = min(ay, frame.shape[0] - 5)
    cv2.drawMarker(frame, (ax, ay), _RED, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 12, _RED, 1, cv2.LINE_AA)
    _shadowed_text(frame, "AIM", (ax + 16, ay + 4), scale=0.45, color=_RED, thickness=1)


def _draw_fire_overlay(frame, zone, trail, laser_xy) -> None:
    from macbook.sweep import draw_fire_overlay
    draw_fire_overlay(frame, zone, trail, laser_xy)


def _draw_fire_status(frame, decision) -> None:
    from macbook.sweep import draw_fire_status
    draw_fire_status(frame, decision, text_fn=_shadowed_text)


def _draw_status_panel(
    frame: np.ndarray,
    motion_det: Detection | None,
    yolo_det: Detection | None,
    yolo_class: str | None,
    fused: Detection | None,
    fps: float,
    yolo_ms: float,
    yolo_skipped: bool,
    frame_idx: int,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 290
    x0 = w - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 10, y0 + 180), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + 180), _WHITE, 1)

    y = y0 + 22
    _shadowed_text(frame, "JETSON LIVE DETECT", (x0 + 12, y), scale=0.55, color=_WHITE, thickness=1)
    y += 22
    _shadowed_text(
        frame,
        f"frame {frame_idx}   {fps:5.1f} fps",
        (x0 + 12, y), scale=0.45, color=(180, 180, 180), thickness=1,
    )
    y += 22
    if motion_det is not None:
        _shadowed_text(frame, f"MOTION  {_fmt_pct(motion_det.confidence)}  a={motion_det.area}",
                       (x0 + 12, y), scale=0.5, color=_CYAN, thickness=1)
    else:
        _shadowed_text(frame, "MOTION  —", (x0 + 12, y), scale=0.5, color=_DIM, thickness=1)
    y += 22

    yolo_tag = "(skipped)" if yolo_skipped else f"({yolo_ms:.0f} ms)"
    _shadowed_text(frame, f"YOLO  {yolo_tag}", (x0 + 12, y), scale=0.5, color=_GREEN, thickness=1)
    y += 18
    if yolo_det is not None and yolo_class is not None:
        _shadowed_text(
            frame, f"  {yolo_class:<14s} {_fmt_pct(yolo_det.confidence)}",
            (x0 + 12, y), scale=0.45, color=_WHITE, thickness=1,
        )
    else:
        _shadowed_text(frame, "  no detections",
                       (x0 + 12, y), scale=0.45, color=_DIM, thickness=1)
    y += 22

    if fused is not None:
        col = _RED if fused.source == "ensemble" else (_GREEN if fused.source == "yolo" else _CYAN)
        _shadowed_text(frame, f"FUSED ({fused.source}) {_fmt_pct(fused.confidence)}",
                       (x0 + 12, y), scale=0.5, color=col, thickness=1)
    else:
        _shadowed_text(frame, "FUSED  —", (x0 + 12, y), scale=0.5, color=_DIM, thickness=1)


def _detect_yolo(
    yolo_inst: YoloDetector,
    frame: np.ndarray,
    conf_threshold: float,
) -> tuple[Detection | None, str | None]:
    """Run YOLO on the full frame; return top detection + class name."""
    results = yolo_inst.model(frame, verbose=False, conf=conf_threshold)
    if not results:
        return None, None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    best_i = max(range(len(boxes)), key=lambda i: float(boxes[i].conf[0]))
    box = boxes[best_i]
    conf = float(box.conf[0])
    if conf < conf_threshold:
        return None, None
    cls_id = int(box.cls[0])
    name = yolo_inst.model.names.get(cls_id, str(cls_id))
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
    det = Detection(
        bbox=(x1, y1, x2, y2),
        centroid=((x1 + x2) // 2, (y1 + y2) // 2),
        confidence=conf,
        source="yolo",
    )
    return det, name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt",
                        help="YOLO weights path (default: project root yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--motion-min-area", type=int, default=400)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO entirely")
    parser.add_argument("--drone-motion", action="store_true",
                        help="Use drone-tuned motion detector (geometric filters + "
                             "persistence tracking) instead of legacy biggest-blob.")
    parser.add_argument("--motion-max-area", type=int, default=20_000,
                        help="Cap blob area in px2; rejects close-up hands/faces "
                             "(default 20k = ~140x140 px box)")
    parser.add_argument("--motion-min-hits", type=int, default=2,
                        help="Min consecutive matches to emit (default 2 = ~1 frame latency)")
    parser.add_argument("--motion-max-misses", type=int, default=10,
                        help="Drop a track after this many missed frames (forgiving for hover)")
    parser.add_argument("--motion-min-aspect", type=float, default=0.25)
    parser.add_argument("--motion-max-aspect", type=float, default=4.0)
    parser.add_argument("--motion-min-solidity", type=float, default=0.40)
    parser.add_argument("--motion-learning-rate", type=float, default=0.005)
    parser.add_argument("--motion-min-speed", type=float, default=12.0,
                        help="Speed (px/frame) required to GRAB the lock. "
                             "Default 12 = ~360 px/sec at 30 fps.")
    parser.add_argument("--motion-demote-speed", type=float, default=3.0,
                        help="Speed (px/frame) below which the lock is RELEASED. "
                             "Hysteresis with --motion-min-speed.")
    parser.add_argument("--motion-min-straightness", type=float, default=0.45,
                        help="Trajectory linearity required to PROMOTE a track. "
                             "1.0=perfectly linear (drone), 0=jitter-in-place "
                             "(face turn, MOG2 noise wobble).")
    parser.add_argument("--hover-lock", action="store_true",
                        help="Once a track is acquired, keep it locked via an "
                             "OpenCV CSRT appearance tracker even after the drone "
                             "stops moving (and MOG2 absorbs it as background).")
    parser.add_argument("--hover-max-s", type=float, default=4.0,
                        help="Max seconds to keep a hover lock without any motion "
                             "re-acquisition before giving up.")
    # --- Laser sweep / FIRE visualization ---
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
    parser.add_argument("--fire-trail-len", type=int, default=120,
                        help="Number of recent aim points to render as the fading trail. "
                             "120 ~ 4s of trail at 30 fps; segments older than ~2s "
                             "are alpha-faded out by the renderer regardless.")
    parser.add_argument("--fire-arm-conf", type=float, default=0.45,
                        help="Detection conf required (sustained) to ARM the laser. "
                             "Lowered from 0.55 because we now arm on the fused "
                             "DETECTION's own confidence instead of the predictor's "
                             "harsh multiplicative score.")
    parser.add_argument("--fire-arm-dwell-s", type=float, default=0.30,
                        help="Seconds of sustained high conf before ARMING. "
                             "Lowered from 0.8s -- with detection-confidence gating, "
                             "0.3s is enough to filter single-frame YOLO blips while "
                             "still firing within ~9 frames of acquisition.")
    parser.add_argument("--fire-disarm-conf", type=float, default=0.20,
                        help="Conf below which an ARMED lock disarms (hysteresis).")
    parser.add_argument("--fire-override", default="auto",
                        choices=["auto", "force_on", "force_off"],
                        help="Manual override for the FIRE controller.")
    parser.add_argument(
        "--lock-filter", action=argparse.BooleanOptionalAction, default=True,
        help="Block sudden centroid teleports between frames so the laser "
             "doesn't chase target swaps. Default: ON.",
    )
    parser.add_argument("--lock-max-jump-px", type=float, default=250.0,
                        help="Pixel distance above which a frame-to-frame centroid "
                             "change is treated as a candidate target swap.")
    parser.add_argument("--lock-cand-frames", type=int, default=5,
                        help="Frames the new candidate must remain stable before "
                             "the swap is accepted (and predictor/FIRE reset).")
    parser.add_argument("--motion-no-sticky", action="store_true",
                        help="Disable track stickiness; pick the highest-scoring "
                             "track every frame (default = sticky)")
    parser.add_argument("--motion-prefer-top", action="store_true",
                        help="Bias toward upper-half-of-frame tracks")
    parser.add_argument("--motion-prefer-bigger", action="store_true",
                        help="On equal speed/persistence prefer LARGER blob "
                             "(default = smaller; the score primary is speed regardless)")
    parser.add_argument("--lead-time", type=float, default=0.0,
                        help="TargetPredictor lead seconds (0.0 = no leading)")
    parser.add_argument("--cable-offset-x", type=float, default=0.0)
    parser.add_argument("--cable-offset-y", type=float, default=0.0)
    parser.add_argument("--yolo-every", type=int, default=1,
                        help="Run YOLO every N frames (default 1; bump to 3-5 if CPU-bound)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, help="Requested camera fps")
    parser.add_argument("--index", type=int, default=0, help="USB camera index")
    parser.add_argument("--csi", action="store_true",
                        help="Use the CSI ribbon camera via nvarguscamerasrc")
    parser.add_argument("--gst", default="",
                        help="Custom GStreamer pipeline string (overrides --csi/--index)")
    parser.add_argument("--headless", action="store_true",
                        help="No window; write rotating jpg + log over SSH")
    parser.add_argument("--snapshot-path", type=Path,
                        default=PROJECT_ROOT / "recordings" / "jetson_live_latest.jpg",
                        help="Where to write the latest annotated frame in --headless mode")
    parser.add_argument("--log-every-s", type=float, default=1.0,
                        help="In --headless mode, print a status line every N seconds")
    parser.add_argument("--mjpeg", action="store_true",
                        help="Serve a real-time MJPEG stream over HTTP. View at "
                             "http://<jetson-ip>:<port>/ in any browser. "
                             "Implies --headless (no local cv2 window).")
    parser.add_argument("--mjpeg-port", type=int, default=8765,
                        help="Port for the MJPEG HTTP stream (default 8765)")
    parser.add_argument("--mjpeg-quality", type=int, default=80,
                        help="JPEG quality 1-100 for the MJPEG stream (default 80)")
    parser.add_argument("--mjpeg-max-fps", type=float, default=30.0,
                        help="Cap MJPEG frame rate (default 30)")
    parser.add_argument("--record", type=Path, default=None,
                        help="Always-on recording: write annotated frames to this "
                             "MP4 file from start to exit. Use the browser REC "
                             "button (when --mjpeg) for on-demand recording instead.")
    parser.add_argument("--record-dir", type=Path,
                        default=PROJECT_ROOT / "recordings",
                        help="Directory for browser-triggered recordings "
                             "(default ./recordings/). Files are auto-named "
                             "jetson_live_YYYYMMDD_HHMMSS.mp4")
    args = parser.parse_args()
    if args.mjpeg:
        # MJPEG implies headless: no local window competing for resources.
        args.headless = True

    if not args.headless and not os.environ.get("DISPLAY"):
        print("WARNING: $DISPLAY is not set. Live window will fail.")
        print("         Either run with --headless, or SSH with `ssh -X cask`,")
        print("         or run directly on a screen attached to the Jetson.")

    cap = _open_camera(args)
    if not cap.isOpened():
        print("FAIL: could not open camera. Try --index 1, --csi, or --gst '...'")
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"Camera reports {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

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
        print(f"Hover-lock ENABLED (CSRT tracker; max {args.hover_max_s}s without motion)")

    from macbook.tracker import TargetPredictor, cable_aim_offset
    predictor: TargetPredictor | None = None
    if args.lead_time > 0 or args.cable_offset_x != 0 or args.cable_offset_y != 0:
        predictor = TargetPredictor(lead_time_s=max(args.lead_time, 1e-6))
        print(f"Using TargetPredictor lead={args.lead_time}s "
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
              f"arm@conf={args.fire_arm_conf} dwell={args.fire_arm_dwell_s}s "
              f"override={args.fire_override}")
    yolo_inst: YoloDetector | None = None
    if not args.no_yolo:
        if not args.weights.exists():
            print(f"FAIL: YOLO weights not found at {args.weights}")
            cap.release()
            return 1
        print(f"Loading YOLO weights from {args.weights}...")
        yolo_inst = YoloDetector(weights=str(args.weights), conf_threshold=args.conf)
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            print(f"YOLO ready ({len(yolo_inst.model.names)} classes)  "
                  f"CUDA={'YES' if cuda_ok else 'NO (CPU fallback)'}")
        except Exception:
            print(f"YOLO ready ({len(yolo_inst.model.names)} classes)")

    streamer = None
    if args.mjpeg:
        from macbook.streaming import MJPEGServer
        streamer = MJPEGServer(
            port=args.mjpeg_port,
            jpeg_quality=args.mjpeg_quality,
            max_fps=args.mjpeg_max_fps,
        )
        streamer.start()
        print(f"MJPEG stream:  http://<jetson-ip>:{args.mjpeg_port}/  "
              f"(quality={args.mjpeg_quality} max_fps={args.mjpeg_max_fps})")

    if args.headless:
        args.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if not streamer:
            print(f"Headless mode: writing latest frame to {args.snapshot_path}")
    else:
        cv2.namedWindow("Katena Live  (q quit, s snap, h help)", cv2.WINDOW_NORMAL)

    fps_window: deque[float] = deque(maxlen=30)
    fps_window.append(time.perf_counter())
    yolo_class_counter: Counter[str] = Counter()
    motion_hits = 0
    fused_hits = 0
    last_log_at = time.perf_counter()
    last_yolo_det: Detection | None = None
    last_yolo_class: str | None = None
    yolo_ms_smoothed = 0.0
    show_help = True
    frame_idx = 0

    # ---- Recording state -------------------------------------------------
    # `record_writer` is non-None while we're actively writing frames.
    # Two ways to be active:
    #   1. --record PATH was given on the CLI (always-on; opened below).
    #   2. The browser REC button toggled streamer.recording_requested True.
    # The producer (this loop) owns the cv2.VideoWriter so the streamer
    # thread doesn't have to deal with codec/dim setup. We poll
    # streamer.recording_requested each frame to decide open/close.
    record_writer: cv2.VideoWriter | None = None
    # Recent laser aim points for the fading-trail overlay. We keep a
    # rolling deque so render cost stays constant regardless of run
    # length. Each item is (t_seconds, (px, py)).
    laser_trail: deque[tuple[float, tuple[float, float]]] = deque(
        maxlen=args.fire_trail_len if args.fire else 1
    )

    record_writer_path: Path | None = None
    record_force_on = False
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record_force_on = True  # opens lazily once we have first-frame dims
    if args.mjpeg:
        args.record_dir.mkdir(parents=True, exist_ok=True)

    def _open_writer(path: Path, fps_hint: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
        h, w = frame_shape[:2]
        # mp4v is the most-compatible codec available everywhere cv2
        # is built. Quality is OK for testing; demo capture is fine.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Use the camera's measured fps if reasonable, else 30.
        fps_safe = float(fps_hint) if 5.0 <= float(fps_hint) <= 120.0 else 30.0
        return cv2.VideoWriter(str(path), fourcc, fps_safe, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break
            frame_idx += 1

            motion_det = motion.detect(frame)
            if motion_det is not None:
                motion_hits += 1

            run_yolo_now = (
                yolo_inst is not None
                and (frame_idx % max(1, args.yolo_every) == 0)
            )
            yolo_skipped = (yolo_inst is not None) and not run_yolo_now
            if run_yolo_now:
                t_y0 = time.perf_counter()
                last_yolo_det, last_yolo_class = _detect_yolo(yolo_inst, frame, args.conf)
                yolo_ms = (time.perf_counter() - t_y0) * 1000.0
                yolo_ms_smoothed = (0.8 * yolo_ms_smoothed + 0.2 * yolo_ms) if yolo_ms_smoothed else yolo_ms
                if last_yolo_class is not None:
                    yolo_class_counter[last_yolo_class] += 1

            yolo_det = last_yolo_det
            yolo_class = last_yolo_class
            fused = _fuse(motion_det, yolo_det)
            if fused is not None:
                fused_hits += 1

            _draw_motion_box(frame, motion_det)
            _draw_yolo_box(frame, yolo_det, yolo_class or "yolo")

            # Stability filter: blocks aim teleports when the fused
            # detection swaps to a new target. During the suspicion
            # window (`fused_for_aim is None`) the predictor will time
            # out and the FIRE state machine will fall through to
            # COOLDOWN, so the laser stays parked instead of slewing.
            if lock_filter is not None:
                fr = lock_filter.update(fused)
                if fr.discontinuity:
                    if predictor is not None:
                        predictor.reset()
                    if fire_ctl is not None:
                        fire_ctl.reset()
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
                    h, w = frame.shape[:2]
                    if 0 <= aim_x_i < w and 0 <= aim_y_i < h:
                        cv2.drawMarker(frame, (aim_x_i, aim_y_i),
                                       (0, 165, 255), cv2.MARKER_DIAMOND, 28, 2, cv2.LINE_AA)
                        cv2.line(frame, (cx, cy), (aim_x_i, aim_y_i),
                                 (0, 165, 255), 1, cv2.LINE_AA)
                        # Show predictor confidence next to the aim diamond.
                        _shadowed_text(
                            frame,
                            f"pred {int(round(tracked.confidence * 100))}%  "
                            f"v={int(tracked.speed_px_per_s)}px/s",
                            (aim_x_i + 18, aim_y_i + 5),
                            scale=0.45, thickness=1,
                        )
                elif predictor.is_active:
                    predictor.reset()

            # ---- FIRE: sweep planning + render -------------------------
            fire_decision = None
            if sweep_planner is not None and fire_ctl is not None:
                # Use the fused detection's own confidence (already a
                # composite "drone-likelihood") instead of the predictor's
                # multiplicative score, which was so harsh that auto-arm
                # almost never triggered for hovering / curving drones.
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
                    laser_xy = sweep_planner.aim_point(time.perf_counter(), zone)
                    laser_trail.append((time.perf_counter(), laser_xy))
                # Always render whatever trail we have. The renderer
                # age-fades old segments, so during cooldown the arc
                # gracefully fades to nothing instead of being wiped on
                # the very next frame.
                _draw_fire_overlay(frame, zone, laser_trail, laser_xy)
                _draw_fire_status(frame, fire_decision)

            now = time.perf_counter()
            fps_window.append(now)
            fps = (
                (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
                if len(fps_window) > 1 else 0.0
            )

            _draw_status_panel(
                frame, motion_det, yolo_det, yolo_class, fused,
                fps, yolo_ms_smoothed, yolo_skipped, frame_idx,
            )
            if not args.headless and show_help:
                _shadowed_text(
                    frame,
                    "q quit   s snapshot   h hide help",
                    (12, frame.shape[0] - 12),
                    scale=0.45, thickness=1,
                )

            if streamer is not None:
                streamer.publish(frame)

            # ---- Recording management (browser-driven OR --record CLI) --
            want_record = record_force_on or (
                streamer is not None and streamer.recording_requested
            )
            if want_record and record_writer is None:
                if record_force_on and args.record is not None:
                    record_writer_path = args.record
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    record_writer_path = args.record_dir / f"jetson_live_{ts}.mp4"
                record_writer = _open_writer(
                    record_writer_path, actual_fps, frame.shape
                )
                if not record_writer.isOpened():
                    print(f"WARN: failed to open recorder at {record_writer_path}")
                    record_writer = None
                    record_writer_path = None
                    if streamer is not None:
                        streamer.set_recording_active(None)
                else:
                    print(f"REC start -> {record_writer_path}")
                    if streamer is not None:
                        streamer.set_recording_active(str(record_writer_path))
            elif (not want_record) and record_writer is not None:
                record_writer.release()
                print(f"REC stop  -> {record_writer_path}")
                record_writer = None
                if streamer is not None:
                    streamer.set_recording_active(None)
                record_writer_path = None
            if record_writer is not None:
                record_writer.write(frame)
            if args.headless and streamer is None:
                ok_enc, buf = cv2.imencode(".jpg", frame)
                if ok_enc:
                    tmp_path = args.snapshot_path.with_suffix(args.snapshot_path.suffix + ".tmp")
                    tmp_path.write_bytes(buf.tobytes())
                    tmp_path.replace(args.snapshot_path)
            if args.headless:
                if (now - last_log_at) >= args.log_every_s:
                    last_log_at = now
                    top_yolo = yolo_class_counter.most_common(3)
                    top_str = "  ".join(f"{n}={c}" for n, c in top_yolo) or "—"
                    # Surface chosen-track diagnostics so we can tune live.
                    if motion_det is not None:
                        track_info = f"area={motion_det.area:5d}"
                        if hasattr(motion, "_preferred_track_id") and \
                           motion._preferred_track_id is not None:
                            for t in motion.all_tracks:
                                if t.track_id == motion._preferred_track_id:
                                    track_info = (
                                        f"speed={t.smoothed_speed_px_per_frame:5.1f}px/f "
                                        f"straight={t.straightness:.2f} "
                                        f"area={t.area:5d}  hits={t.hits:3d}"
                                    )
                                    break
                    else:
                        track_info = "(no detection)"
                    pred_info = ""
                    if tracked is not None:
                        pred_info = f"  pred_conf={tracked.confidence:.2f}"
                    print(
                        f"frame={frame_idx:6d}  fps={fps:5.1f}  "
                        f"motion_hits={motion_hits:6d}  {track_info}{pred_info}  "
                        f"yolo: {top_str}",
                        flush=True,
                    )
            else:
                cv2.imshow("Katena Live  (q quit, s snap, h help)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    snap = PROJECT_ROOT / "recordings" / f"jetson_live_snap_{frame_idx:06d}.jpg"
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap), frame)
                    print(f"Saved {snap}")
                elif key == ord("h"):
                    show_help = not show_help

    finally:
        cap.release()
        if record_writer is not None:
            record_writer.release()
            print(f"REC stop  -> {record_writer_path}")
            if streamer is not None:
                streamer.set_recording_active(None)
        if not args.headless:
            cv2.destroyAllWindows()

    print()
    print("=== SUMMARY ===")
    print(f"Frames processed:   {frame_idx}")
    print(f"Motion hits:        {motion_hits}")
    print(f"Fused detections:   {fused_hits}")
    if yolo_class_counter:
        print("Top YOLO classes:")
        for name, cnt in yolo_class_counter.most_common(10):
            print(f"  {name:<20s} {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
