"""Centralized configuration. Reads .env once, exposes typed accessors.

Usage:
    from macbook.config import settings
    cap = cv2.VideoCapture(settings.camera_index)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    foundry_url: str = ""
    foundry_token: str = ""
    foundry_ontology_rid: str = ""
    foundry_osdk_client_id: str = ""
    foundry_osdk_client_secret: str = ""

    pico_serial_port: str = "/dev/cu.usbmodem101"
    pico_baudrate: int = 115200

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    jetson_host: str = ""
    jetson_user: str = ""
    jetson_ssh_key: str = "~/.ssh/id_ed25519"

    default_sweep_radius_deg: float = 3.0
    default_sweep_hz: float = 0.5
    fiber_signal_threshold: float = 0.20

    engagements_log_path: Path = PROJECT_ROOT / "engagements.jsonl"
    calibration_path: Path = PROJECT_ROOT / "calibration.json"
    yolo_weights_path: Path = PROJECT_ROOT / "yolov8n.pt"
    recordings_dir: Path = PROJECT_ROOT / "recordings"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        foundry_url=_env("FOUNDRY_URL"),
        foundry_token=_env("FOUNDRY_TOKEN"),
        foundry_ontology_rid=_env("FOUNDRY_ONTOLOGY_RID"),
        foundry_osdk_client_id=_env("FOUNDRY_OSDK_CLIENT_ID"),
        foundry_osdk_client_secret=_env("FOUNDRY_OSDK_CLIENT_SECRET"),
        pico_serial_port=_env("PICO_SERIAL_PORT", "/dev/cu.usbmodem101"),
        pico_baudrate=_env_int("PICO_BAUDRATE", 115200),
        camera_index=_env_int("CAMERA_INDEX", 0),
        camera_width=_env_int("CAMERA_WIDTH", 1280),
        camera_height=_env_int("CAMERA_HEIGHT", 720),
        jetson_host=_env("JETSON_HOST"),
        jetson_user=_env("JETSON_USER"),
        jetson_ssh_key=_env("JETSON_SSH_KEY", "~/.ssh/id_ed25519"),
        default_sweep_radius_deg=_env_float("DEFAULT_SWEEP_RADIUS_DEG", 3.0),
        default_sweep_hz=_env_float("DEFAULT_SWEEP_HZ", 0.5),
        fiber_signal_threshold=_env_float("FIBER_SIGNAL_THRESHOLD", 0.20),
    )


settings = get_settings()
