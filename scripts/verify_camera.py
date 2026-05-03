"""Quick camera smoke test.

Opens the default camera, reads a frame, prints the resolution.
If the macOS camera permission dialog hasn't been answered yet, this
will trigger it. Grant permission in System Settings > Privacy &
Security > Camera, then quit and reopen Cursor and run this again.

Usage:
    python scripts/verify_camera.py
    python scripts/verify_camera.py --index 1     # try a different camera
    python scripts/verify_camera.py --preview     # show a live window
"""

import argparse
import sys
import time

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a live preview window (press q to quit)",
    )
    args = parser.parse_args()

    print(f"Opening camera index {args.index}...")
    cap = cv2.VideoCapture(args.index)

    if not cap.isOpened():
        print("FAIL: could not open camera.")
        print("  - On macOS: System Settings > Privacy & Security > Camera,")
        print("    enable Cursor (or your terminal), then restart it.")
        print("  - Try a different --index (1, 2, ...) for external USB cams.")
        return 1

    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        print("FAIL: camera opened but frame read returned no data.")
        cap.release()
        return 1

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"OK: camera index {args.index} delivered {w}x{h} @ {fps:.1f} fps reported")

    if args.preview:
        print("Live preview — press q in the window to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("verify_camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
