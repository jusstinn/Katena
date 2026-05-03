.PHONY: help test test-fast test-verbose test-cov lint smoke clean dashboard tracker calibrate seed sync-jetson replay download-drone-data train-drone train-drone-bg train-status train-stop

PY := .venv/bin/python
ACTIVATE := source .venv/bin/activate

help:
	@echo "Katena — common commands"
	@echo ""
	@echo "  make test           Run unit tests (fast, no hardware)"
	@echo "  make test-fast      Run tests, fail on first error"
	@echo "  make test-verbose   Run tests with full output"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make smoke          Run scripts/verify_all.py"
	@echo ""
	@echo "  make tracker        Run main_tracker (mock Pico)"
	@echo "  make calibrate      Run calibration_tool (mock Pico)"
	@echo "  make dashboard      Open Streamlit dashboard"
	@echo "  make seed           Seed engagements.jsonl with demo data"
	@echo "  make replay VIDEO=path/to/clip.mp4   Replay a video through the detector stack"
	@echo ""
	@echo "  make download-drone-data    Fetch Seraphim drone YOLO dataset from HuggingFace"
	@echo "  make train-drone            Fine-tune yolov8n on drone dataset (foreground)"
	@echo "  make train-drone-bg         Same, but in the background; logs to logs/train.log"
	@echo "  make train-status           Tail the latest training log"
	@echo "  make train-stop             Stop the background training run"
	@echo ""
	@echo "  make sync-jetson    rsync project to JETSON_HOST (set in .env)"
	@echo "  make clean          Remove caches and __pycache__"

test:
	@$(PY) -m pytest

test-fast:
	@$(PY) -m pytest -x

test-verbose:
	@$(PY) -m pytest -v

test-cov:
	@$(PY) -m pytest --cov --cov-report=term-missing

smoke:
	@$(PY) scripts/verify_all.py

tracker:
	@$(PY) -m macbook.main_tracker --mock

calibrate:
	@$(PY) -m macbook.calibration_tool --mock

dashboard:
	@$(ACTIVATE) && streamlit run dashboard.py

seed:
	@$(PY) scripts/seed_engagements.py --reset --count 8

replay:
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make replay VIDEO=path/to/clip.mp4 [OPTS=\"--save --all-yolo\"]"; exit 1; fi
	@$(PY) scripts/replay_video.py $(VIDEO) $(OPTS)

# --- Drone YOLO training -----------------------------------------------------
DATA ?= datasets/seraphim_drone/data.yaml
TRAIN_OPTS ?=

download-drone-data:
	@$(PY) scripts/download_drone_dataset.py $(DL_OPTS)

train-drone:
	@$(PY) scripts/train_drone_yolo.py --data $(DATA) $(TRAIN_OPTS)

train-drone-bg:
	@mkdir -p logs
	@if pgrep -f train_drone_yolo.py >/dev/null; then \
		echo "Training is already running (PID $$(pgrep -f train_drone_yolo.py)). Use 'make train-status' or 'make train-stop'."; exit 1; \
	fi
	@nohup $(PY) -u scripts/train_drone_yolo.py --data $(DATA) $(TRAIN_OPTS) > logs/train.log 2>&1 & echo $$! > logs/train.pid
	@sleep 1
	@echo "Training started (PID $$(cat logs/train.pid)). Tail with:"
	@echo "  make train-status   # or: tail -f logs/train.log"

train-status:
	@if [ ! -f logs/train.pid ]; then echo "No background training run recorded."; exit 1; fi
	@PID=$$(cat logs/train.pid); \
	if kill -0 $$PID 2>/dev/null; then echo "[running] PID $$PID"; else echo "[stopped] PID $$PID exited"; fi
	@echo "--- last 40 log lines ---"
	@tail -n 40 logs/train.log

train-stop:
	@if [ -f logs/train.pid ]; then \
		PID=$$(cat logs/train.pid); kill $$PID 2>/dev/null && echo "Sent SIGTERM to $$PID" || echo "PID $$PID was not running"; \
		rm -f logs/train.pid; \
	else \
		pkill -f train_drone_yolo.py && echo "Killed train_drone_yolo.py processes" || echo "Nothing to kill"; \
	fi

sync-jetson:
	@if [ -z "$$JETSON_HOST" ]; then echo "Set JETSON_HOST in .env first."; exit 1; fi
	@rsync -av --exclude='.venv*' --exclude='downloads' --exclude='__pycache__' \
		--exclude='.git' --exclude='recordings' \
		./ $${JETSON_USER}@$${JETSON_HOST}:~/Katena/

clean:
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "Cleaned caches."
