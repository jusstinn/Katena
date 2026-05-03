"""Tests for the config module — env var precedence and defaults."""

from __future__ import annotations

import importlib

import pytest


def _reload_config():
    import macbook.config as cfg
    importlib.reload(cfg)
    return cfg


class TestConfig:
    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch):
        for k in (
            "FOUNDRY_URL", "FOUNDRY_TOKEN", "PICO_SERIAL_PORT", "PICO_BAUDRATE",
            "CAMERA_INDEX", "CAMERA_WIDTH", "DEFAULT_SWEEP_RADIUS_DEG",
        ):
            monkeypatch.delenv(k, raising=False)
        cfg = _reload_config()
        s = cfg.get_settings()
        assert s.foundry_url == ""
        assert s.pico_baudrate == 115200
        assert s.camera_index == 0
        assert s.default_sweep_radius_deg == 3.0

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FOUNDRY_URL", "https://example.com")
        monkeypatch.setenv("PICO_BAUDRATE", "230400")
        monkeypatch.setenv("CAMERA_INDEX", "2")
        monkeypatch.setenv("DEFAULT_SWEEP_RADIUS_DEG", "5.5")
        cfg = _reload_config()
        s = cfg.get_settings()
        assert s.foundry_url == "https://example.com"
        assert s.pico_baudrate == 230400
        assert s.camera_index == 2
        assert s.default_sweep_radius_deg == 5.5

    def test_paths_under_project_root(self):
        cfg = _reload_config()
        s = cfg.get_settings()
        assert s.engagements_log_path.name == "engagements.jsonl"
        assert s.calibration_path.name == "calibration.json"
        assert s.yolo_weights_path.name == "yolov8n.pt"
        assert s.engagements_log_path.parent == s.project_root
