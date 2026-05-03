"""YOLOv8 smoke test.

Loads yolov8n.pt, runs a single inference on a synthetic frame (or a
live camera frame with --camera), and reports timing + detected classes.

Out-of-the-box COCO has 80 classes — none of them is "drone". You will
see things like person, bottle, laptop. That's expected and confirms
the inference pipeline is healthy. Drone detection requires fine-tuning
(see jetson/setup_jetson.sh and macbook/ — coming next).

Usage:
    python scripts/verify_yolo.py
    python scripts/verify_yolo.py --camera     # one frame from camera
    python scripts/verify_yolo.py --preview    # live YOLO overlay
"""

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--camera", action="store_true", help="Single inference on a camera frame")
    parser.add_argument("--preview", action="store_true", help="Live overlay window (q to quit)")
    parser.add_argument("--index", type=int, default=0, help="Camera index for --camera/--preview")
    args = parser.parse_args()

    print(f"Loading {args.weights}...")
    model = YOLO(args.weights)
    print(f"OK: loaded {args.weights} with {len(model.names)} classes")

    if args.preview:
        cap = cv2.VideoCapture(args.index)
        if not cap.isOpened():
            print("FAIL: camera not available. Run scripts/verify_camera.py first.")
            return 1
        print("Live YOLO preview — press q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("verify_yolo (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return 0

    if args.camera:
        cap = cv2.VideoCapture(args.index)
        if not cap.isOpened():
            print("FAIL: camera not available.")
            return 1
        time.sleep(0.5)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("FAIL: camera read returned no frame.")
            return 1
    else:
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    t0 = time.perf_counter()
    results = model(frame, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    detections = results[0].boxes
    print(f"OK: inference took {elapsed_ms:.1f} ms on {frame.shape[1]}x{frame.shape[0]} frame")
    if len(detections) == 0:
        print("  No detections (expected for a synthetic gray frame).")
    else:
        print(f"  {len(detections)} detection(s):")
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f"    - {name} (conf={conf:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
