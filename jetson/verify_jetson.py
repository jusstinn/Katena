"""Jetson hardware + driver smoke test.

Run on the CASK / Jetson to verify:
  - Jetson model + JetPack version
  - GPU / CUDA availability
  - OpenCV CUDA support
  - PyTorch CUDA availability
  - All cameras (CSI ribbon + USB)
  - Disk / memory headroom

Usage:
    python3 jetson/verify_jetson.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def section(title: str) -> None:
    bar = "=" * (len(title) + 8)
    print(f"\n{bar}\n=== {title} ===\n{bar}")


def jetson_info() -> None:
    section("JETSON")
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        print(rel.read_text().strip().splitlines()[0])
    else:
        print("/etc/nv_tegra_release missing — not a Jetson?")
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        try:
            print(f"Model: {model_file.read_text().strip(chr(0))}")
        except Exception as e:
            print(f"Model: unreadable ({e})")


def cuda_info() -> None:
    section("CUDA / GPU")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
            print(out.strip())
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
    elif shutil.which("tegrastats"):
        print("nvidia-smi not present (typical on Jetson). tegrastats is available.")
    else:
        print("Neither nvidia-smi nor tegrastats found.")


def python_libs() -> None:
    section("PYTHON STACK")
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
        try:
            n = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"  CUDA-enabled OpenCV devices: {n}")
        except Exception:
            print("  cv2.cuda not present (OpenCV built without CUDA)")
    except ImportError as e:
        print(f"OpenCV: MISSING ({e})")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"PyTorch: MISSING ({e})")
    try:
        from ultralytics import YOLO  # noqa: F401
        print("Ultralytics: importable")
    except ImportError as e:
        print(f"Ultralytics: MISSING ({e})")


def cameras() -> None:
    section("CAMERAS")
    try:
        import cv2
    except ImportError:
        print("OpenCV missing — skipping camera probe.")
        return

    found_any = False
    for index in range(0, 4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  index {index}: OK   {w}x{h}")
                found_any = True
            else:
                print(f"  index {index}: opened but no frame")
            cap.release()
        else:
            print(f"  index {index}: not available")
    if not found_any:
        print("\nNo cameras delivered frames. If using CSI ribbon, you may need")
        print("a GStreamer pipeline. Try:")
        print('  cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), '
              'width=1280, height=720, framerate=30/1 ! nvvidconv ! '
              'video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)')


def resources() -> None:
    section("RESOURCES")
    try:
        import shutil as sh
        total, used, free = sh.disk_usage("/")
        gb = 1024 ** 3
        print(f"Disk:  {used / gb:6.1f}G used / {total / gb:6.1f}G total ({free / gb:.1f}G free)")
    except Exception as e:
        print(f"Disk: {e}")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    print(line.strip())
    except Exception as e:
        print(f"Mem: {e}")


def main() -> int:
    jetson_info()
    cuda_info()
    python_libs()
    cameras()
    resources()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
