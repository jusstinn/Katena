"""Camera frame overlay — state machine UI, bbox, status bar.

This is what makes the demo feel like a real ops system rather than a
debug window. Drawn on every tracker frame:

  - Big state banner (top-center) with the current EngagementState
  - System status bar (top-right): camera / serial / foundry / sdr dots
  - Detection bbox + centroid + aim point crosshair
  - Latency/FPS in bottom-left
  - Fiber signal level bar in bottom-right
  - Optional pre-recorded video frame insert (for demo handoff)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection
from .state_machine import EngagementState, EngagementStateMachine

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_DIM = (120, 120, 120)
_GREEN = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 220)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class SystemStatus:
    """Health of each subsystem. Drawn as colored dots in the status bar."""

    camera: bool = False
    serial: bool = False
    foundry: bool = False
    sdr: bool | None = None  # None = not in use, True/False = state

    extras: dict[str, bool | None] = field(default_factory=dict)


def _shadowed_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = _WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), _FONT, scale, _BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_state_banner(frame: np.ndarray, sm: EngagementStateMachine) -> None:
    """Big state text centered at the top."""
    text = sm.state.value
    color = sm.color()
    h, w = frame.shape[:2]
    scale = max(0.8, min(1.6, w / 900.0))
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x = (w - tw) // 2
    y = th + 25
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 16, 5), (x + tw + 16, y + 12), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x - 16, 5), (x + tw + 16, y + 12), color, 2)
    _shadowed_text(frame, text, (x, y), scale=scale, color=color, thickness=thickness)


def draw_status_bar(frame: np.ndarray, status: SystemStatus) -> None:
    """Colored dots top-right: camera / serial / foundry / sdr."""
    h, w = frame.shape[:2]
    items: list[tuple[str, bool | None]] = [
        ("CAM", status.camera),
        ("PICO", status.serial),
        ("FNDY", status.foundry),
    ]
    if status.sdr is not None:
        items.append(("SDR", status.sdr))
    for name, val in status.extras.items():
        items.append((name, val))

    pad = 10
    radius = 6
    spacing = 70
    y = 30
    x = w - pad - spacing * len(items)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 12, 8), (w - pad + 4, 52), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (name, val) in enumerate(items):
        cx = x + i * spacing + radius
        if val is True:
            color = _GREEN
        elif val is False:
            color = _RED
        else:
            color = _DIM
        cv2.circle(frame, (cx, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, y), radius, _BLACK, 1, cv2.LINE_AA)
        _shadowed_text(frame, name, (cx + radius + 4, y + 4), scale=0.45, thickness=1)


def draw_detection(
    frame: np.ndarray,
    det: Detection,
    state_color: tuple[int, int, int],
    drop_fraction: float = 0.15,
) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
    label = f"{det.source} {det.confidence:.2f}"
    _shadowed_text(frame, label, (x1, max(15, y1 - 6)), scale=0.5, thickness=1)
    cv2.circle(frame, det.centroid, 4, state_color, -1, cv2.LINE_AA)

    ax, ay = det.aim_point(drop_fraction)
    ay = min(ay, frame.shape[0] - 5)
    cv2.drawMarker(frame, (ax, ay), _RED, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 12, _RED, 1, cv2.LINE_AA)
    _shadowed_text(frame, "AIM", (ax + 16, ay + 4), scale=0.45, color=_RED, thickness=1)


def draw_telemetry(
    frame: np.ndarray,
    fps: float,
    detection_ms: float | None = None,
    aim_ms: float | None = None,
) -> None:
    """Bottom-left: latency numbers."""
    h = frame.shape[0]
    lines = [f"FPS: {fps:5.1f}"]
    if detection_ms is not None:
        lines.append(f"Detect: {detection_ms:5.1f} ms")
    if aim_ms is not None:
        lines.append(f"Aim:    {aim_ms:5.1f} ms")
    y = h - 12 - 18 * (len(lines) - 1)
    for line in lines:
        _shadowed_text(frame, line, (12, y), scale=0.5, thickness=1)
        y += 18


def draw_signal_bar(
    frame: np.ndarray,
    signal: float,
    threshold: float = 0.2,
    label: str = "FIBER SIGNAL",
) -> None:
    """Bottom-right: fiber signal level bar (1.0 = full, 0.0 = severed)."""
    h, w = frame.shape[:2]
    bar_w = 220
    bar_h = 18
    x1 = w - bar_w - 12
    y1 = h - bar_h - 12
    x2 = x1 + bar_w
    y2 = y1 + bar_h
    cv2.rectangle(frame, (x1 - 1, y1 - 18), (x2 + 1, y2 + 1), _BLACK, -1)
    fill = int(bar_w * max(0.0, min(1.0, signal)))
    color = _GREEN if signal > 0.6 else (_YELLOW if signal > threshold else _RED)
    cv2.rectangle(frame, (x1, y1), (x1 + fill, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _WHITE, 1)
    _shadowed_text(frame, f"{label} {int(signal * 100):3d}%", (x1, y1 - 4), scale=0.45, thickness=1)


def draw_crosshair(frame: np.ndarray, point: tuple[int, int] | None = None) -> None:
    """Center crosshair (or at a custom point) for calibration UI."""
    h, w = frame.shape[:2]
    cx, cy = point or (w // 2, h // 2)
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), _WHITE, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), _WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 18, _WHITE, 1, cv2.LINE_AA)


def render_full(
    frame: np.ndarray,
    sm: EngagementStateMachine,
    status: SystemStatus,
    detection: Detection | None,
    fps: float,
    detection_ms: float | None = None,
    aim_ms: float | None = None,
    fiber_signal: float = 1.0,
    fiber_threshold: float = 0.2,
    drop_fraction: float = 0.15,
) -> np.ndarray:
    """Convenience: draw everything in one call. Returns the frame (mutated)."""
    if detection is not None:
        draw_detection(frame, detection, sm.color(), drop_fraction=drop_fraction)
    draw_state_banner(frame, sm)
    draw_status_bar(frame, status)
    draw_telemetry(frame, fps, detection_ms=detection_ms, aim_ms=aim_ms)
    draw_signal_bar(frame, fiber_signal, threshold=fiber_threshold)
    return frame
