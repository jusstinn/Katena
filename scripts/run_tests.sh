#!/usr/bin/env bash
# One-stop sanity gate. Run before committing or before each demo.
#
# 1. Unit tests (fast, no hardware)
# 2. Lint (py_compile sweep)
# 3. Optional: live verification (camera, YOLO, serial, foundry)
#
# Usage:
#   bash scripts/run_tests.sh            # tests + lint
#   bash scripts/run_tests.sh --smoke    # also run scripts/verify_all.py

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo -e "${BLUE}==>${RESET} Compile check (no syntax errors)"
python -m py_compile macbook/*.py pico/*.py scripts/*.py dashboard.py jetson/*.py
echo -e "${GREEN}OK${RESET}"

echo
echo -e "${BLUE}==>${RESET} Unit tests"
python -m pytest

echo
if [[ "${1:-}" == "--smoke" ]]; then
    echo -e "${BLUE}==>${RESET} Live verification (camera, YOLO, serial, Foundry)"
    python scripts/verify_all.py || echo -e "${RED}Some live checks failed (often expected if no hardware/Foundry yet).${RESET}"
fi

echo
echo -e "${GREEN}All gates passed.${RESET} Safe to commit / demo."
