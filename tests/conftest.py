"""Shared pytest fixtures + the MicroPython `machine` shim.

The `machine` shim lets us import `pico/pico_controller.py` from CPython
(the firmware uses `from machine import Pin, PWM, ADC` which only exists
on the actual Pico). We replace the module with stubs that record calls
so we can inspect what the firmware *would* have done with real hardware.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _FakePin:
    OUT = 1
    IN = 0
    PULL_UP = 2
    PULL_DOWN = 3

    def __init__(self, pin_no, mode=None, pull=None):
        self.pin_no = pin_no
        self.mode = mode
        self.pull = pull
        self._value = 0
        self.history: list[int] = []

    def value(self, v=None):
        if v is None:
            return self._value
        self._value = int(v)
        self.history.append(self._value)


class _FakePWM:
    def __init__(self, pin):
        self.pin = pin
        self._freq = 50
        self._duty = 0
        self.duty_history: list[int] = []

    def freq(self, f=None):
        if f is None:
            return self._freq
        self._freq = f

    def duty_u16(self, d=None):
        if d is None:
            return self._duty
        self._duty = int(d)
        self.duty_history.append(self._duty)


class _FakeADC:
    def __init__(self, channel):
        self.channel = channel
        self._value = 32_768

    def read_u16(self):
        return self._value


def _install_machine_shim() -> None:
    if "machine" in sys.modules:
        return
    mod = types.ModuleType("machine")
    mod.Pin = _FakePin
    mod.PWM = _FakePWM
    mod.ADC = _FakeADC
    sys.modules["machine"] = mod


_install_machine_shim()


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "engagements.jsonl"


@pytest.fixture
def tmp_calibration(tmp_path: Path) -> Path:
    return tmp_path / "calibration.json"


@pytest.fixture
def gray_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def moving_blob_frames() -> list[np.ndarray]:
    """A 30-frame sequence with a 60x40 white rectangle sliding across.

    Useful for exercising the MOG2 background subtractor — needs several
    frames of "background" before it can identify motion as foreground.
    """
    import cv2

    frames = []
    for i in range(30):
        f = np.full((480, 640, 3), 60, dtype=np.uint8)
        if i > 4:
            x = 100 + i * 12
            y = 200
            cv2.rectangle(f, (x - 30, y - 20), (x + 30, y + 20), (220, 220, 220), -1)
        frames.append(f)
    return frames


@pytest.fixture
def pico_controller():
    """Import the Pico firmware module (with `machine` shim already installed)."""
    sys.path.insert(0, str(PROJECT_ROOT / "pico"))
    import pico_controller  # noqa: PLC0415

    return pico_controller
