.PHONY: help test test-fast test-verbose test-cov lint smoke clean dashboard tracker calibrate seed sync-jetson

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

sync-jetson:
	@if [ -z "$$JETSON_HOST" ]; then echo "Set JETSON_HOST in .env first."; exit 1; fi
	@rsync -av --exclude='.venv*' --exclude='downloads' --exclude='__pycache__' \
		--exclude='.git' --exclude='recordings' \
		./ $${JETSON_USER}@$${JETSON_HOST}:~/Katena/

clean:
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "Cleaned caches."
