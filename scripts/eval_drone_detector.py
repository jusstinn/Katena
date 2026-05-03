#!/usr/bin/env python3
"""Evaluate a drone-detector YOLO model against a video, with sanity checks.

Why this exists:
    Counting "frames with a 'drone' box" lies. A model that locks onto a
    ceiling fixture for the whole clip looks 100% accurate on paper.

What this script measures (per video, per weights file):
    detection_rate        % frames with at least one drone box >= conf
    centroid_std_x/y      stdev of top-box center over all detection frames
                          (low values => the box barely moves => stuck)
    max_stuck_run         longest run of consecutive frames where the top
                          centroid stayed within `--stuck-px` (default 8 px)
                          for >= `--stuck-min` frames (default 30)
    stuck_run_count       number of such stuck runs
    motion_overlap_rate   % of detection frames where the YOLO top-box has
                          IoU > 0 with the motion-detector blob; this is
                          our best proxy for "the YOLO box is on the
                          actually-moving thing in the frame".
    area_mean/std         bbox area statistics (stuck => low std)
    class_distribution    histogram of top-box classes seen

Output:
    Pretty stdout summary AND a JSON report at <video>.<weights_stem>.eval.json

Usage:
    python scripts/eval_drone_detector.py recordings/IMG_2019.MOV \
        --weights models/hf/aeroyolo_n.pt --class drone
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from macbook.detector import MotionDetector  # noqa: E402


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Min YOLO confidence to count as a detection")
    p.add_argument("--class", dest="cls", default=None,
                   help="If set, only count boxes whose class name matches "
                        "(case-insensitive). Multi-class models like aeroyolo "
                        "should pass --class drone.")
    p.add_argument("--stuck-px", type=int, default=8,
                   help="Centroid jitter (pixels) below which we call a frame 'stuck'")
    p.add_argument("--stuck-min", type=int, default=30,
                   help="A stuck-run must last at least this many frames to count")
    p.add_argument("--motion-min-area", type=int, default=400)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--report", type=Path, default=None,
                   help="Where to write the JSON report (default: alongside video)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"FAIL: video not found: {args.video}")
        return 1
    if not args.weights.exists():
        print(f"FAIL: weights not found: {args.weights}")
        return 1

    from ultralytics import YOLO  # local import: keeps --help fast
    print(f"Loading {args.weights} ...")
    model = YOLO(str(args.weights))
    names: dict[int, str] = dict(model.names)
    print(f"  classes: {list(names.values())}")
    cls_filter = args.cls.lower() if args.cls else None
    if cls_filter and cls_filter not in {n.lower() for n in names.values()}:
        print(f"WARN: --class '{args.cls}' not in model classes; will match nothing.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"FAIL: cannot open {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.video.name}  {width}x{height} @ {fps:.1f} fps  "
          f"({total} frames)")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    motion = MotionDetector(min_area=args.motion_min_area)

    per_frame: list[dict] = []
    class_counter: Counter[str] = Counter()

    t_start = time.perf_counter()
    frame_idx = args.start_frame
    motion_only_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and (frame_idx - args.start_frame) >= args.max_frames:
            break

        m_det = motion.detect(frame)
        m_bbox = m_det.bbox if m_det else None

        results = model(frame, verbose=False, conf=args.conf)
        boxes = results[0].boxes if results else None

        top = None  # (cls_name, conf, bbox, area)
        if boxes is not None and len(boxes) > 0:
            best_conf = -1.0
            for i in range(len(boxes)):
                conf = float(boxes[i].conf[0])
                cls_id = int(boxes[i].cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                if cls_filter and cls_name.lower() != cls_filter:
                    continue
                if conf > best_conf:
                    x1, y1, x2, y2 = (int(v) for v in boxes[i].xyxy[0].tolist())
                    top = (cls_name, conf, (x1, y1, x2, y2),
                           max(0, x2 - x1) * max(0, y2 - y1))
                    best_conf = conf

        if top is not None:
            cls_name, conf, bbox, area = top
            class_counter[cls_name] += 1
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            iou = _bbox_iou(bbox, m_bbox) if m_bbox else 0.0
            per_frame.append({
                "f": frame_idx,
                "cls": cls_name,
                "conf": round(conf, 3),
                "cx": cx,
                "cy": cy,
                "area": area,
                "motion_iou": round(iou, 3),
                "has_motion": m_bbox is not None,
            })
        else:
            if m_bbox is not None:
                motion_only_frames += 1

        frame_idx += 1
        if frame_idx % 60 == 0:
            elapsed = time.perf_counter() - t_start
            done = frame_idx - args.start_frame
            print(f"  frame {frame_idx}/{total}  "
                  f"({done / elapsed:.1f} fps inference)")

    cap.release()
    elapsed = time.perf_counter() - t_start
    processed = frame_idx - args.start_frame
    detected = len(per_frame)

    centroids_x = [d["cx"] for d in per_frame]
    centroids_y = [d["cy"] for d in per_frame]
    areas = [d["area"] for d in per_frame]

    def _safe_std(xs: list[float]) -> float:
        return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0

    def _safe_mean(xs: list[float]) -> float:
        return float(statistics.fmean(xs)) if xs else 0.0

    stuck_runs: list[int] = []
    if len(per_frame) >= args.stuck_min:
        run_start = 0
        for i in range(1, len(per_frame)):
            dx = per_frame[i]["cx"] - per_frame[run_start]["cx"]
            dy = per_frame[i]["cy"] - per_frame[run_start]["cy"]
            if (dx * dx + dy * dy) ** 0.5 > args.stuck_px:
                length = i - run_start
                if length >= args.stuck_min:
                    stuck_runs.append(length)
                run_start = i
        tail = len(per_frame) - run_start
        if tail >= args.stuck_min:
            stuck_runs.append(tail)

    motion_overlap_count = sum(1 for d in per_frame if d["motion_iou"] > 0.0)
    overlap_rate = (motion_overlap_count / detected) if detected else 0.0

    summary = {
        "video": str(args.video),
        "weights": str(args.weights),
        "class_filter": args.cls,
        "frames_total": total,
        "frames_processed": processed,
        "frames_with_detection": detected,
        "detection_rate": round(detected / processed, 3) if processed else 0.0,
        "frames_motion_only": motion_only_frames,
        "class_distribution": dict(class_counter.most_common()),
        "centroid_std_x_px": round(_safe_std(centroids_x), 2),
        "centroid_std_y_px": round(_safe_std(centroids_y), 2),
        "stuck_runs_count": len(stuck_runs),
        "stuck_runs_total_frames": sum(stuck_runs),
        "stuck_run_max_frames": max(stuck_runs) if stuck_runs else 0,
        "stuck_px_threshold": args.stuck_px,
        "stuck_min_frames": args.stuck_min,
        "motion_overlap_rate": round(overlap_rate, 3),
        "motion_overlap_frames": motion_overlap_count,
        "area_mean_px2": round(_safe_mean(areas), 1),
        "area_std_px2": round(_safe_std(areas), 1),
        "inference_seconds": round(elapsed, 1),
        "inference_fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
    }

    report_path = args.report or args.video.with_name(
        f"{args.video.stem}.{args.weights.stem}.eval.json"
    )
    report_path.write_text(json.dumps({"summary": summary, "per_frame": per_frame},
                                      indent=2))

    print()
    print("=== EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<28s} {v}")
    print(f"\nFull report: {report_path}")

    if detected and overlap_rate < 0.10:
        print("\n  WARNING: motion_overlap_rate is very low. The YOLO box rarely "
              "lands on the moving object — the model is probably stuck on a "
              "static distractor (e.g. ceiling fixture).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
