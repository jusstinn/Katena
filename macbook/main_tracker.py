"""Main live tracking pipeline — the centerpiece.

Orchestrates everything:
    Camera -> Detector -> StateMachine -> Calibration -> PicoLink
                                       \\-> Logger
                                       \\-> Overlay -> on-screen

State machine progression (auto-engage mode, the demo flow):

    SEARCHING (no detection)
        |  (detection appears)
        v
    TARGET_ACQUIRED (~1s of stable tracking)
        |
        v
    CLASSIFYING (~1s, optional SDR RF-silence check)
        |
        v
    FOG_CONFIRMED (~0.5s)
        |
        v
    ENGAGING (servos lock on, mode=LOCKED, fiber signal monitoring)
        |  (signal drops below 50%)
        v
    FIBER_COMPROMISED
        |  (signal drops below threshold from .env)
        v
    TARGET_NEUTRALIZED (~3s celebration, then back to SEARCHING)

Keyboard:
    Space   manual advance (only matters with --no-auto-engage)
    X       STAND_DOWN (emergency abort)
    R       reset to SEARCHING
    H       toggle help overlay
    Q/Esc   quit

Usage:
    python -m macbook.main_tracker
    python -m macbook.main_tracker --mock              # no Pico hardware
    python -m macbook.main_tracker --no-yolo           # motion only
    python -m macbook.main_tracker --no-auto-engage    # require Space to advance
    python -m macbook.main_tracker --camera 1 --weights yolov8n.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from .calibration import Calibration
from .config import settings
from .detector import Detection, EnsembleDetector, MotionDetector, YoloDetector
from .engagement import DroneEngagement, ThreatLevel
from .logger import DualLogger, FoundryOSDKLogger, LocalJSONLogger, next_drone_id
from .overlay import SystemStatus, _shadowed_text, render_full
from .serial_link import PicoMode, open_link
from .state_machine import EngagementState, EngagementStateMachine

DWELL_ACQUIRED_S = 1.0
DWELL_CLASSIFYING_S = 1.0
DWELL_FOG_CONFIRMED_S = 0.5
DWELL_NEUTRALIZED_S = 3.0
TARGET_LOST_TIMEOUT_S = 0.7
FIBER_COMPROMISED_THRESHOLD = 0.5


class Tracker:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        cal: Calibration,
        use_yolo: bool,
        yolo_weights: Path,
        mock_pico: bool,
        pico_port: str,
        auto_engage: bool,
        engagements_log: Path,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        self.cal = cal
        motion = MotionDetector(min_area=400)
        yolo = YoloDetector(weights=str(yolo_weights), conf_threshold=0.25) if use_yolo else None
        self.detector = EnsembleDetector(motion=motion, yolo=yolo)

        self.link = open_link(mock=mock_pico, port=pico_port, verbose=False)
        self.logger = DualLogger(LocalJSONLogger(engagements_log), FoundryOSDKLogger())

        self.sm = EngagementStateMachine()
        self.sm.on_change(self._on_state_change)

        self.auto_engage = auto_engage
        self.show_help = True
        self.window = "Katena Tracker  (h help, q quit)"

        self.current_engagement: DroneEngagement | None = None
        self.state_entered_at = time.time()
        self.last_detection_at = 0.0

        self._fps_window: deque[float] = deque(maxlen=30)
        self._frame_t0 = time.perf_counter()

    def _on_state_change(self, prev: EngagementState, new: EngagementState) -> None:
        self.state_entered_at = time.time()
        eng = self.current_engagement
        if eng is None:
            return
        if new == EngagementState.ENGAGING:
            eng.mark_engagement_started()
            self.logger.update(eng)
        elif new == EngagementState.FIBER_COMPROMISED:
            self.logger.update(eng)
        elif new == EngagementState.TARGET_NEUTRALIZED:
            eng.fiber_cut = True
            eng.signal_lost = True
            eng.mark_engagement_ended()
            self.logger.update(eng)
        elif new == EngagementState.TARGET_LOST:
            eng.notes = "lost"
            self.logger.update(eng)
        elif new == EngagementState.STAND_DOWN:
            eng.notes = "stand down"
            self.logger.update(eng)

    def _state_dwell(self) -> float:
        return time.time() - self.state_entered_at

    def _command_aim(self, det: Detection, mode: PicoMode) -> tuple[float, float]:
        ax, ay = det.aim_point()
        ax = max(0, min(self.cal.frame_width - 1, ax))
        ay = max(0, min(self.cal.frame_height - 1, ay))
        pan, tilt = self.cal.pixel_to_servo(float(ax), float(ay))
        self.link.aim(pan, tilt, mode)
        return pan, tilt

    def _advance(self, det: Detection | None, fiber_signal: float) -> None:
        state = self.sm.state

        if state == EngagementState.SEARCHING:
            if det is not None:
                self.current_engagement = DroneEngagement(
                    drone_id=next_drone_id(),
                    threat_level=ThreatLevel.HIGH,
                    sensor_fusion=["camera"],
                )
                self.logger.create(self.current_engagement)
                self.sm.transition(EngagementState.TARGET_ACQUIRED)

        elif state == EngagementState.TARGET_ACQUIRED:
            if det is None and self._state_dwell() > TARGET_LOST_TIMEOUT_S:
                self.sm.transition(EngagementState.TARGET_LOST)
            elif self.auto_engage and self._state_dwell() >= DWELL_ACQUIRED_S:
                self.sm.transition(EngagementState.CLASSIFYING)

        elif state == EngagementState.CLASSIFYING:
            if det is None and self._state_dwell() > TARGET_LOST_TIMEOUT_S:
                self.sm.transition(EngagementState.TARGET_LOST)
            elif self.auto_engage and self._state_dwell() >= DWELL_CLASSIFYING_S:
                if self.current_engagement is not None:
                    self.current_engagement.rf_silence_confirmed = True
                    if "rf_silence" not in self.current_engagement.sensor_fusion:
                        self.current_engagement.sensor_fusion.append("rf_silence")
                    self.logger.update(self.current_engagement)
                self.sm.transition(EngagementState.FOG_CONFIRMED)

        elif state == EngagementState.FOG_CONFIRMED:
            if self.auto_engage and self._state_dwell() >= DWELL_FOG_CONFIRMED_S:
                self.sm.transition(EngagementState.ENGAGING)

        elif state == EngagementState.ENGAGING:
            if fiber_signal < FIBER_COMPROMISED_THRESHOLD:
                self.sm.transition(EngagementState.FIBER_COMPROMISED)

        elif state == EngagementState.FIBER_COMPROMISED:
            if fiber_signal < settings.fiber_signal_threshold:
                self.sm.transition(EngagementState.TARGET_NEUTRALIZED)

        elif state == EngagementState.TARGET_NEUTRALIZED:
            if self._state_dwell() >= DWELL_NEUTRALIZED_S:
                self.current_engagement = None
                self.sm.transition(EngagementState.SEARCHING)

        elif state in (EngagementState.TARGET_LOST, EngagementState.STAND_DOWN):
            if self._state_dwell() >= 1.5:
                self.current_engagement = None
                self.sm.transition(EngagementState.SEARCHING)

    def _drive_servos(self, det: Detection | None) -> tuple[float | None, float | None]:
        state = self.sm.state
        if det is None:
            return None, None
        if state in (EngagementState.TARGET_ACQUIRED, EngagementState.CLASSIFYING, EngagementState.FOG_CONFIRMED):
            pan, tilt = self._command_aim(det, PicoMode.TRACKING)
            return pan, tilt
        if state in (EngagementState.ENGAGING, EngagementState.FIBER_COMPROMISED):
            pan, tilt = self._command_aim(det, PicoMode.LOCKED)
            if self.current_engagement is not None:
                self.current_engagement.pan_angle = pan
                self.current_engagement.tilt_angle = tilt
            return pan, tilt
        return None, None

    def _draw_help(self, frame) -> None:
        if not self.show_help:
            return
        lines = [
            "Space: manual advance    X: STAND DOWN    R: reset    H: hide",
            f"auto-engage: {self.auto_engage}    yolo: {self.detector.yolo is not None}    pico: {self.link.is_connected()}",
        ]
        for i, line in enumerate(lines):
            _shadowed_text(frame, line, (12, frame.shape[0] - 70 + i * 18), scale=0.45, thickness=1)

    def run(self) -> int:
        cv2.namedWindow(self.window)
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read failed.")
                    return 1

                t_det0 = time.perf_counter()
                det = self.detector.detect(frame)
                detection_ms = (time.perf_counter() - t_det0) * 1000.0

                if det is not None:
                    self.last_detection_at = time.time()
                    if self.current_engagement is not None:
                        self.current_engagement.position_x = float(det.centroid[0])
                        self.current_engagement.position_y = float(det.centroid[1])

                tel = self.link.telemetry()
                fiber_signal = tel.fiber_signal

                t_aim0 = time.perf_counter()
                self._drive_servos(det)
                aim_ms = (time.perf_counter() - t_aim0) * 1000.0

                self._advance(det, fiber_signal)

                if self.current_engagement is not None:
                    self.current_engagement.signal_strength = fiber_signal
                    self.current_engagement.signal_lost = fiber_signal < settings.fiber_signal_threshold

                now = time.perf_counter()
                self._fps_window.append(now)
                fps = 0.0
                if len(self._fps_window) > 1:
                    fps = (len(self._fps_window) - 1) / (self._fps_window[-1] - self._fps_window[0])

                status = SystemStatus(
                    camera=True,
                    serial=self.link.is_connected(),
                    foundry=False,
                    sdr=None,
                )
                render_full(
                    frame, self.sm, status, det,
                    fps=fps,
                    detection_ms=detection_ms,
                    aim_ms=aim_ms,
                    fiber_signal=fiber_signal,
                    fiber_threshold=settings.fiber_signal_threshold,
                )
                self._draw_help(frame)
                cv2.imshow(self.window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("h"):
                    self.show_help = not self.show_help
                elif key == ord("x"):
                    self.sm.transition(EngagementState.STAND_DOWN)
                elif key == ord("r"):
                    self.sm.reset()
                elif key == ord(" ") and not self.auto_engage:
                    next_states = {
                        EngagementState.TARGET_ACQUIRED: EngagementState.CLASSIFYING,
                        EngagementState.CLASSIFYING: EngagementState.FOG_CONFIRMED,
                        EngagementState.FOG_CONFIRMED: EngagementState.ENGAGING,
                    }
                    nxt = next_states.get(self.sm.state)
                    if nxt is not None:
                        self.sm.transition(nxt)
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
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO branch (motion only)")
    parser.add_argument("--weights", type=Path, default=settings.yolo_weights_path)
    parser.add_argument("--no-auto-engage", action="store_true", help="Require Space to advance state")
    parser.add_argument("--log", type=Path, default=settings.engagements_log_path)
    args = parser.parse_args()

    cal = Calibration.load(args.cal)
    if not cal.anchors:
        print(f"Warning: no calibration loaded from {args.cal}.")
        print("         Aim will fall back to servo center (90,90).")
        print("         Run `python -m macbook.calibration_tool --mock` first.")

    tracker = Tracker(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        cal=cal,
        use_yolo=not args.no_yolo,
        yolo_weights=args.weights,
        mock_pico=args.mock,
        pico_port=args.port,
        auto_engage=not args.no_auto_engage,
        engagements_log=args.log,
    )
    return tracker.run()


if __name__ == "__main__":
    sys.exit(main())
