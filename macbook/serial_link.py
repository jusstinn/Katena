"""Pico W serial bridge.

Two implementations behind one interface:
  - RealPicoLink:  pyserial connection to the Pico, background thread
                   parses telemetry lines.
  - MockPicoLink:  no hardware needed. Prints commands to stdout and
                   simulates telemetry — including a fiber-severance
                   simulation so the full demo runs end-to-end without
                   any physical photodetector.

Protocol:
    Host -> Pico:   "P{pan}T{tilt}M{mode}\n"   (pan/tilt 0-180, mode 0-3)
    Pico -> Host:   "D{cm}S{status}L{ldr}\n"   (distance, status, LDR 0-1023)
"""

from __future__ import annotations

import math
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import serial


class PicoMode(IntEnum):
    IDLE = 0
    TRACKING = 1
    SWEEP = 2
    LOCKED = 3


@dataclass
class PicoTelemetry:
    """Latest reading from the Pico. distance/ldr may be None if no sensor."""

    distance_cm: float | None = None
    status: int = 0
    ldr_raw: int | None = None
    received_at: float = 0.0
    fiber_signal: float = 1.0


def _format_command(pan: float, tilt: float, mode: PicoMode) -> bytes:
    pan = max(0.0, min(180.0, pan))
    tilt = max(0.0, min(180.0, tilt))
    return f"P{pan:.1f}T{tilt:.1f}M{int(mode)}\n".encode()


_TELEMETRY_RE = re.compile(r"D(?P<d>-?\d+(?:\.\d+)?)S(?P<s>\d+)L(?P<l>\d+)")


def _parse_telemetry(line: str, ldr_full: int = 900, ldr_cut: int = 50) -> PicoTelemetry | None:
    m = _TELEMETRY_RE.search(line)
    if not m:
        return None
    distance = float(m.group("d"))
    status = int(m.group("s"))
    ldr = int(m.group("l"))
    span = max(1, ldr_full - ldr_cut)
    signal = max(0.0, min(1.0, (ldr - ldr_cut) / span))
    return PicoTelemetry(
        distance_cm=distance if distance >= 0 else None,
        status=status,
        ldr_raw=ldr,
        received_at=time.time(),
        fiber_signal=signal,
    )


class PicoLink(ABC):
    @abstractmethod
    def aim(self, pan: float, tilt: float, mode: PicoMode = PicoMode.TRACKING) -> None: ...

    @abstractmethod
    def telemetry(self) -> PicoTelemetry: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    def close(self) -> None:
        return None


class RealPicoLink(PicoLink):
    """Talks to a real Pico W over USB serial. Background thread reads telemetry."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        read_timeout: float = 0.2,
        ldr_full: int = 900,
        ldr_cut: int = 50,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.ldr_full = ldr_full
        self.ldr_cut = ldr_cut
        self._serial: serial.Serial | None = None
        self._latest = PicoTelemetry()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        try:
            self._serial = serial.Serial(port, baudrate, timeout=read_timeout)
            time.sleep(0.5)
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
        except serial.SerialException:
            self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def aim(self, pan: float, tilt: float, mode: PicoMode = PicoMode.TRACKING) -> None:
        if not self.is_connected():
            return
        cmd = _format_command(pan, tilt, mode)
        try:
            assert self._serial is not None
            self._serial.write(cmd)
        except serial.SerialException:
            self._serial = None

    def telemetry(self) -> PicoTelemetry:
        with self._lock:
            return self._latest

    def _read_loop(self) -> None:
        assert self._serial is not None
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = self._serial.read(64)
            except serial.SerialException:
                break
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                try:
                    text = line.decode(errors="ignore").strip()
                except Exception:
                    continue
                tel = _parse_telemetry(text, self.ldr_full, self.ldr_cut)
                if tel:
                    with self._lock:
                        self._latest = tel

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


class MockPicoLink(PicoLink):
    """No hardware needed. Logs commands and simulates fiber severance.

    Behavior:
      - Default fiber signal stays at 1.0
      - When mode == LOCKED, signal degrades over `severance_seconds`
      - After it reaches 0, stays at 0 (drone neutralized)
      - Distance jitters around `mock_distance_cm` to simulate ultrasonic noise
      - All commands echoed to stdout if `verbose` is True
    """

    def __init__(
        self,
        verbose: bool = False,
        mock_distance_cm: float = 150.0,
        severance_seconds: float = 4.0,
    ) -> None:
        self.verbose = verbose
        self.mock_distance_cm = mock_distance_cm
        self.severance_seconds = severance_seconds
        self._engage_started_at: float | None = None
        self._signal_floor = 1.0
        self._mode = PicoMode.IDLE
        self._last_pan = 90.0
        self._last_tilt = 90.0

    def is_connected(self) -> bool:
        return True

    def aim(self, pan: float, tilt: float, mode: PicoMode = PicoMode.TRACKING) -> None:
        self._last_pan = pan
        self._last_tilt = tilt
        prev = self._mode
        self._mode = mode
        if mode == PicoMode.LOCKED and prev != PicoMode.LOCKED:
            self._engage_started_at = time.time()
        if mode != PicoMode.LOCKED:
            self._engage_started_at = None
        if self.verbose:
            print(f"[mock-pico] P{pan:6.1f} T{tilt:6.1f} M{int(mode)}")

    def telemetry(self) -> PicoTelemetry:
        signal = self._signal_floor
        if self._mode == PicoMode.LOCKED and self._engage_started_at is not None:
            elapsed = time.time() - self._engage_started_at
            t = min(1.0, elapsed / self.severance_seconds)
            signal = max(0.0, 1.0 - t)
            self._signal_floor = min(self._signal_floor, signal)
        elif self._mode != PicoMode.LOCKED:
            self._signal_floor = min(1.0, self._signal_floor + 0.001)
            signal = self._signal_floor

        jitter = math.sin(time.time() * 7) * 1.5
        distance = self.mock_distance_cm + jitter
        ldr_raw = int(50 + signal * (900 - 50))

        return PicoTelemetry(
            distance_cm=distance,
            status=int(self._mode),
            ldr_raw=ldr_raw,
            received_at=time.time(),
            fiber_signal=signal,
        )

    def reset_severance(self) -> None:
        self._signal_floor = 1.0
        self._engage_started_at = None


def open_link(
    mock: bool = False,
    port: str | None = None,
    baudrate: int = 115200,
    **mock_kwargs: Any,
) -> PicoLink:
    """Factory: returns a real link if available, else falls back to mock if requested.

    Pass `mock=True` to force mock mode (hardware-less development).
    Otherwise tries to open the real serial port; if that fails (port
    not present), returns a mock link with a warning printed.
    """
    if mock:
        return MockPicoLink(**mock_kwargs)
    if port is None:
        from .config import settings
        port = settings.pico_serial_port
    link = RealPicoLink(port, baudrate)
    if link.is_connected():
        return link
    print(f"[serial_link] Could not open {port} — falling back to mock mode.")
    return MockPicoLink(**mock_kwargs)
