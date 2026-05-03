"""Teleop calibration UI.

Run this once to teach the system how camera pixels map to servo angles.
You jog the servos until the laser dot lands on a known pixel, click
that pixel, repeat for ~5-9 spread-out points. Then the tracker can
turn any pixel into a servo command.

Two interaction modes (toggle with M):

  JOG mode (default)  - WASD nudges servos by 1 deg (5 with Shift).
                        LEFT-CLICK adds an anchor at the clicked pixel
                        with the current servo position.
                        Use this to build the calibration map.

  TEST mode           - LEFT-CLICK uses the current calibration to
                        command the servos to the clicked pixel. Verify
                        the laser dot lands on the click point.

Other keys:
    1-9     jump pan/tilt to preset positions
    s       save calibration to disk
    l       (re)load calibration from disk
    c       clear all anchors
    r       reset servos to center (90/90)
    z       remove the nearest anchor to the mouse cursor
    h       toggle help overlay
    q / Esc quit

Usage:
    python -m macbook.calibration_tool
    python -m macbook.calibration_tool --mock        # use mock Pico
    python -m macbook.calibration_tool --camera 1    # external USB cam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, CalibrationAnchor
from .config import settings
from .overlay import _shadowed_text, draw_crosshair, draw_status_bar, SystemStatus
from .serial_link import PicoMode, open_link

JOG_STEP_DEG = 1.0
JOG_STEP_DEG_SHIFT = 5.0


class CalibrationApp:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        cal_path: Path,
        mock: bool,
        port: str,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        self.cal_path = cal_path
        self.cal = Calibration.load(cal_path)
        self.cal.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.cal.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.link = open_link(mock=mock, port=port, verbose=mock)

        self.pan = self.cal.pan_center
        self.tilt = self.cal.tilt_center
        self._command_servos()

        self.mode = "JOG"
        self.show_help = True
        self.mouse_xy: tuple[int, int] = (self.cal.frame_width // 2, self.cal.frame_height // 2)
        self.message: tuple[str, float] = ("", 0.0)
        self.window = "Katena Calibration  (h for help, q to quit)"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _flash(self, msg: str) -> None:
        self.message = (msg, time.time() + 2.5)

    def _command_servos(self) -> None:
        self.link.aim(self.pan, self.tilt, PicoMode.IDLE)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        self.mouse_xy = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode == "JOG":
            self.cal.add(CalibrationAnchor(float(x), float(y), self.pan, self.tilt))
            self._flash(f"Added anchor at ({x},{y}) -> pan={self.pan:.1f} tilt={self.tilt:.1f}")
        else:
            pan, tilt = self.cal.pixel_to_servo(float(x), float(y))
            self.pan = pan
            self.tilt = tilt
            self._command_servos()
            self._flash(f"Aimed at ({x},{y}) -> pan={pan:.1f} tilt={tilt:.1f}")

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False

        shift = key in (ord("W"), ord("A"), ord("S"), ord("D"))
        step = JOG_STEP_DEG_SHIFT if shift else JOG_STEP_DEG
        pan, tilt = self.pan, self.tilt
        if key in (ord("w"), ord("W"), 0):
            tilt -= step
        elif key in (ord("s"), ord("S"), 1):
            tilt += step
        elif key in (ord("a"), ord("A"), 2):
            pan -= step
        elif key in (ord("d"), ord("D"), 3):
            pan += step
        elif key == ord("r"):
            pan, tilt = self.cal.pan_center, self.cal.tilt_center
            self._flash("Reset to center")
        elif key == ord("m"):
            self.mode = "TEST" if self.mode == "JOG" else "JOG"
            self._flash(f"Mode: {self.mode}")
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("s"):
            self.cal.save(self.cal_path)
            self._flash(f"Saved {len(self.cal.anchors)} anchors to {self.cal_path.name}")
        elif key == ord("l"):
            self.cal = Calibration.load(self.cal_path)
            self._flash(f"Loaded {len(self.cal.anchors)} anchors")
        elif key == ord("c"):
            self.cal.anchors.clear()
            self._flash("Cleared all anchors")
        elif key == ord("z"):
            mx, my = self.mouse_xy
            removed = self.cal.remove_nearest(float(mx), float(my))
            self._flash("Removed nearest anchor" if removed else "No anchor near cursor")
        elif ord("1") <= key <= ord("9"):
            preset = int(chr(key))
            grid = [(60, 60), (90, 60), (120, 60), (60, 90), (90, 90), (120, 90),
                    (60, 120), (90, 120), (120, 120)]
            pan, tilt = grid[preset - 1]
            self._flash(f"Preset {preset}: pan={pan} tilt={tilt}")

        pan = max(self.cal.pan_min, min(self.cal.pan_max, pan))
        tilt = max(self.cal.tilt_min, min(self.cal.tilt_max, tilt))
        if pan != self.pan or tilt != self.tilt:
            self.pan, self.tilt = pan, tilt
            self._command_servos()
        return True

    def _draw_overlays(self, frame: np.ndarray) -> None:
        for i, a in enumerate(self.cal.anchors):
            cv2.circle(frame, (int(a.pixel_x), int(a.pixel_y)), 8, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(a.pixel_x), int(a.pixel_y)), 2, (0, 200, 255), -1, cv2.LINE_AA)
            _shadowed_text(
                frame,
                f"{i + 1} ({a.pan_angle:.0f},{a.tilt_angle:.0f})",
                (int(a.pixel_x) + 12, int(a.pixel_y) - 8),
                scale=0.4,
                thickness=1,
            )

        if self.mode == "TEST" and len(self.cal.anchors) >= 2:
            mx, my = self.mouse_xy
            pred_pan, pred_tilt = self.cal.pixel_to_servo(float(mx), float(my))
            _shadowed_text(
                frame,
                f"would aim: pan={pred_pan:.1f} tilt={pred_tilt:.1f}",
                (mx + 12, my + 16),
                scale=0.45,
                thickness=1,
            )
            cv2.drawMarker(frame, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)

        draw_crosshair(frame)

        h, w = frame.shape[:2]
        bar_lines = [
            f"MODE: {self.mode}    PAN: {self.pan:6.1f}    TILT: {self.tilt:6.1f}    ANCHORS: {len(self.cal.anchors)}",
            f"PICO: {'CONNECTED' if self.link.is_connected() else 'OFFLINE'}    FILE: {self.cal_path.name}",
        ]
        y = h - 16 - 18 * (len(bar_lines) - 1)
        for line in bar_lines:
            _shadowed_text(frame, line, (12, y), scale=0.55, thickness=1)
            y += 18

        if self.show_help:
            help_lines = [
                "WASD: nudge 1deg   (capslock = 5deg)",
                "LEFT-CLICK: " + ("add anchor at clicked pixel" if self.mode == "JOG" else "AIM at clicked pixel via calibration"),
                "M: switch JOG/TEST    1-9: presets    R: reset to center",
                "Z: remove nearest anchor    C: clear all",
                "S: save    L: load    H: toggle help    Q/Esc: quit",
            ]
            box_h = 22 * len(help_lines) + 12
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 70), (560, 70 + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            for i, line in enumerate(help_lines):
                _shadowed_text(frame, line, (20, 92 + i * 22), scale=0.5, thickness=1)

        if self.message[1] > time.time():
            _shadowed_text(frame, self.message[0], (12, 50), scale=0.6, color=(0, 255, 255), thickness=2)

        status = SystemStatus(camera=True, serial=self.link.is_connected(), foundry=False)
        draw_status_bar(frame, status)

    def run(self) -> int:
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read failed.")
                    return 1
                self._draw_overlays(frame)
                cv2.imshow(self.window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_key(key):
                        break
        finally:
            self.link.close()
            self.cap.release()
            cv2.destroyAllWindows()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=settings.camera_index)
    parser.add_argument("--width", type=int, default=settings.camera_width)
    parser.add_argument("--height", type=int, default=settings.camera_height)
    parser.add_argument("--cal", type=Path, default=settings.calibration_path)
    parser.add_argument("--mock", action="store_true", help="Force mock Pico (no hardware)")
    parser.add_argument("--port", default=settings.pico_serial_port)
    args = parser.parse_args()

    app = CalibrationApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        cal_path=args.cal,
        mock=args.mock,
        port=args.port,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
