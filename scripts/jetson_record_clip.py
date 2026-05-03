"""Record a short live clip from the Jetson's USB camera and save to /tmp."""

import cv2
import os
import time

OUT_PATH = "/tmp/jetson_live.mp4"
FPS = 20
DURATION_S = 10.0
WIDTH = 1280
HEIGHT = 720


def main() -> int:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("FAIL: camera did not open")
        return 1

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    period = 1.0 / FPS

    start = time.time()
    next_cap = start
    frames = 0
    while time.time() - start < DURATION_S:
        now = time.time()
        if now < next_cap:
            time.sleep(next_cap - now)
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - start
        ts = time.strftime("%H:%M:%S")
        h, w = frame.shape[:2]
        label = f"JETSON LIVE  |  {ts} EDT  |  t={elapsed:4.1f}s  |  {w}x{h}"
        cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.putText(
            frame, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        out.write(frame)
        frames += 1
        next_cap += period

    out.release()
    cap.release()
    elapsed = time.time() - start
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(
        f"WROTE {frames} frames in {elapsed:.1f}s "
        f"({frames / elapsed:.1f} fps), size={size_kb:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
