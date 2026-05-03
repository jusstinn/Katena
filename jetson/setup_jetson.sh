#!/usr/bin/env bash
# Set up the FOG neutralizer environment on a Palantir CASK / NVIDIA Jetson.
#
# Run this on the Jetson AFTER you have transferred the project files
# (e.g. via `git clone` or `scp -r ~/Desktop/Katena jetson:~/`).
#
# Strategy:
#   - Use the SYSTEM Python on the Jetson (do NOT install a new Python
#     via uv/pyenv — Jetson's CUDA-enabled OpenCV and PyTorch are tied
#     to the system Python).
#   - Create a venv with --system-site-packages so it inherits NVIDIA's
#     pre-built OpenCV / PyTorch instead of pulling CPU-only wheels.
#   - Pip-install the lightweight cross-platform deps separately.
#
# Usage:
#   bash jetson/setup_jetson.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Project root: $PROJECT_ROOT"

if [[ ! -f /etc/nv_tegra_release ]]; then
    echo "WARNING: /etc/nv_tegra_release not found. This may not be a Jetson."
    echo "         Continuing anyway — abort with Ctrl-C if this is wrong."
    sleep 2
else
    echo "==> Detected Jetson:"
    head -1 /etc/nv_tegra_release
fi

echo "==> System Python:"
python3 --version
which python3

echo "==> Creating venv (with --system-site-packages to inherit CUDA libs)..."
if [[ ! -d .venv-jetson ]]; then
    python3 -m venv --system-site-packages .venv-jetson
fi
# shellcheck disable=SC1091
source .venv-jetson/bin/activate

echo "==> Upgrading pip / wheel..."
python3 -m pip install --upgrade pip wheel setuptools

echo "==> Installing lightweight Python deps..."
# Notes:
#   - Skipping torch/torchvision: rely on the Jetson's CUDA-enabled build.
#   - Skipping opencv-python: rely on the Jetson's CUDA-enabled cv2.
#   - Ultralytics is OK (it will use the system torch via inheritance).
python3 -m pip install \
    pyserial \
    numpy \
    streamlit \
    pillow \
    requests \
    python-dotenv \
    foundry-platform-sdk \
    ultralytics

echo "==> Verifying inherited CV/ML stack..."
python3 - <<'PY'
import sys
print(f"Python: {sys.version}")
try:
    import cv2; print(f"OpenCV: {cv2.__version__}")
    print(f"  CUDA-enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
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
    from ultralytics import YOLO
    print("Ultralytics: importable")
except ImportError as e:
    print(f"Ultralytics: MISSING ({e})")
try:
    import serial; print(f"PySerial: {serial.__version__}")
except ImportError as e:
    print(f"PySerial: MISSING ({e})")
PY

echo ""
echo "==> Done. Activate the venv with:"
echo "    source $PROJECT_ROOT/.venv-jetson/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Run camera check:   python3 jetson/verify_jetson.py"
echo "  2. Configure .env with FOUNDRY_URL and FOUNDRY_TOKEN"
echo "  3. Run YOLO smoke:     python3 scripts/verify_yolo.py"
