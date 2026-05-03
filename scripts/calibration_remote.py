"""Remote calibration: live MJPEG from Jetson + click-to-anchor on Mac.

The classic ``macbook/calibration_tool.py`` assumed the camera, the
OpenCV preview window, and the Pico were all on the same machine. Our
production rig has them split: the camera + Pico live on the Jetson
(``cask``), and we want to do the actual click-to-aim work on the Mac
because that's where the user is sitting.

This script does exactly that:

  * Pulls the live MJPEG stream from the Jetson's existing MJPEG
    server (the one ``scripts/jetson_live_detect.py --mjpeg`` already
    serves on port 8765 by default).
  * Decodes JPEG frames in a background thread so the cv2 UI stays
    responsive even if the network hiccups.
  * Spawns a long-lived SSH connection to the Jetson that runs
    ``scripts/pico_bridge.py``. Every pan/tilt/jog command is written
    to that SSH session's stdin, which the bridge forwards to
    ``/dev/ttyACM0``. One persistent SSH session, one TTY open --
    no per-keystroke ssh thrash.
  * Mouse clicks add or test calibration anchors EXACTLY like
    ``calibration_tool.py``; same JSON format on disk, so anchors
    captured here drop straight into ``jetson_live_detect.py
    --pico-port ... --cal calibration.json`` without conversion.

Important: while this tool is running, ``jetson_live_detect.py`` on
the Jetson MUST NOT also own the Pico (i.e. don't pass ``--pico-port``
to it). It can keep streaming the camera (--mjpeg), and you almost
certainly want ``--rotate-180`` if your camera is mounted upside
down -- otherwise the anchors you save will be flipped relative to
runtime.

Typical workflow:

  1. On the Jetson:
       python3 scripts/jetson_live_detect.py --mjpeg --rotate-180 \
                                             --no-yolo --motion-min-area 99999

     (The dummy ``--motion-min-area`` and ``--no-yolo`` make it a
     pure camera-feed -> MJPEG passthrough. Skip if you want to see
     detection overlays during calibration too -- not recommended.)

  2. SSH-tunnel the MJPEG port to the Mac:
       ssh -fN -L 8765:127.0.0.1:8765 cask

  3. On the Mac (this script):
       python3 scripts/calibration_remote.py --ssh cask

  4. JOG mode (default): WASD drives the laser. When the dot is on
     a known target, LEFT-CLICK that target's pixel in the preview.
     Repeat 5-9 times across the frame. SHIFT+LEFT-CLICK adds a
     rotation anchor.

  5. M switches to TEST mode: LEFT-CLICK now uses the calibration to
     command the Pico to that pixel. Verify the dot lands where you
     clicked. Add more anchors anywhere it's off.

  6. F saves to ./calibration.json on the Mac. Then push to Jetson:
       scp calibration.json cask:~/Katena/calibration.json

  7. Restart the live detector with closed-loop laser:
       python3 scripts/jetson_live_detect.py --mjpeg --rotate-180 \
                                             --fire --pico-port /dev/ttyACM0 \
                                             --cal ~/Katena/calibration.json

CLI:
  python3 scripts/calibration_remote.py
  python3 scripts/calibration_remote.py --mjpeg http://localhost:8765/stream.mjpg
  python3 scripts/calibration_remote.py --ssh cask --cal ./calibration.json
  python3 scripts/calibration_remote.py --bridge-cmd "python3 /tmp/bridge.py"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from macbook.calibration import (  # noqa: E402
    Calibration,
    CalibrationAnchor,
    RotationAnchor,
)
from macbook.overlay import _shadowed_text, draw_crosshair  # noqa: E402
from macbook.serial_link import PicoMode, _format_command  # noqa: E402

JOG_STEP_DEG = 1.0
JOG_STEP_DEG_SHIFT = 5.0
JOG_ROT_STEP_DEG = 2.0
JOG_ROT_STEP_DEG_SHIFT = 10.0


# ---------------------------------------------------------------------------
# MJPEG ingestion
# ---------------------------------------------------------------------------

class MJPEGSource:
    """Background MJPEG reader.

    Connects to ``url`` (typically the Jetson's existing
    ``http://.../stream.mjpg`` endpoint), parses the multipart stream
    in a worker thread, and exposes the most-recent decoded frame via
    ``read()``. The cv2 main loop never blocks waiting for the network.
    """

    def __init__(self, url: str, *, boundary: bytes = b"--frameboundary",
                 connect_timeout: float = 5.0) -> None:
        self.url = url
        self.boundary = boundary
        self.connect_timeout = connect_timeout
        self._latest: np.ndarray | None = None
        self._latest_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._err: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return False, None, 0
            # Return a copy so the caller can scribble overlays without
            # racing the producer thread.
            return True, self._latest.copy(), self._latest_seq

    def last_error(self) -> str | None:
        return self._err

    # --- internals -------------------------------------------------------

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._stream_once()
                backoff = 0.5
            except Exception as exc:
                self._err = f"{type(exc).__name__}: {exc}"
                # Hold the most-recent frame so the UI doesn't go black on
                # a transient hiccup. Just retry with capped exponential
                # backoff.
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 2, 5.0)

    def _stream_once(self) -> None:
        req = urllib.request.Request(self.url, headers={"User-Agent": "blackfiber-calibration"})
        with urllib.request.urlopen(req, timeout=self.connect_timeout) as resp:
            self._err = None
            buf = b""
            chunk_size = 16 * 1024
            while not self._stop.is_set():
                data = resp.read(chunk_size)
                if not data:
                    raise IOError("server closed stream")
                buf += data
                # Pull out as many complete JPEG frames as live in `buf`.
                while True:
                    # Each frame starts at `boundary\r\n`, then headers,
                    # then \r\n\r\n, then `Content-Length` bytes of JPEG,
                    # then \r\n.
                    bidx = buf.find(self.boundary)
                    if bidx < 0:
                        # No boundary marker yet; need more bytes.
                        break
                    # Headers begin right after the boundary's CRLF.
                    hstart = bidx + len(self.boundary)
                    hend = buf.find(b"\r\n\r\n", hstart)
                    if hend < 0:
                        break  # need more bytes for full headers
                    headers_raw = buf[hstart:hend].decode("ascii", errors="ignore")
                    clen = None
                    for line in headers_raw.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            try:
                                clen = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                clen = None
                            break
                    if clen is None:
                        # Malformed header. Drop everything up to and
                        # including these headers and try to resync on
                        # the next boundary.
                        buf = buf[hend + 4:]
                        break
                    body_start = hend + 4
                    body_end = body_start + clen
                    if body_end > len(buf):
                        break  # need more bytes for the JPEG body
                    jpeg = buf[body_start:body_end]
                    # Trailing CRLF after the JPEG body.
                    buf = buf[body_end + 2:]
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    if arr.size == 0:
                        continue
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    with self._lock:
                        self._latest = img
                        self._latest_seq += 1


# ---------------------------------------------------------------------------
# Pico over SSH
# ---------------------------------------------------------------------------

class PicoBridgeLink:
    """Duck-typed `serial_link.PicoLink` that talks to a remote Pico.

    Spawns ``ssh HOST <bridge cmd>`` once, then forwards each call to
    ``aim()`` as a single line written to that SSH session's stdin.
    The Jetson-side bridge (``scripts/pico_bridge.py``) reads those
    lines and writes them verbatim to ``/dev/ttyACM0``.

    On hardware-less smoke tests pass ``ssh_host=None`` and you'll get
    a no-op link that just logs the commands to stdout.
    """

    def __init__(self, ssh_host: str | None, *, bridge_cmd: str | None,
                 ssh_port: int | None = None, mock_log: bool = False,
                 verbose: bool = False) -> None:
        self.ssh_host = ssh_host
        self.bridge_cmd = bridge_cmd or (
            "python3 ~/Katena/scripts/pico_bridge.py --no-telemetry"
        )
        self.verbose = verbose
        self.mock_log = mock_log
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        if ssh_host is None:
            return
        cmd = ["ssh"]
        if ssh_port:
            cmd += ["-p", str(ssh_port)]
        # ServerAliveInterval keeps the SSH tunnel alive across Wi-Fi
        # blips; -tt forces a TTY so the bridge dies cleanly when SSH
        # disconnects (otherwise it'd linger and hold /dev/ttyACM0).
        cmd += [
            "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3",
            ssh_host,
            self.bridge_cmd,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Tiny grace period; if SSH fails it usually exits within
            # a few hundred ms.
            time.sleep(0.4)
            if self._proc.poll() is not None:
                err = self._proc.stderr.read().decode("utf-8", errors="ignore")
                print(f"[pico_bridge] ssh exited immediately: {err.strip()}")
                self._proc = None
        except FileNotFoundError:
            print("[pico_bridge] 'ssh' not found on PATH")
            self._proc = None

    def is_connected(self) -> bool:
        if self.mock_log:
            return True
        return self._proc is not None and self._proc.poll() is None

    def aim(
        self,
        pan: float,
        tilt: float,
        mode: PicoMode = PicoMode.TRACKING,
        rotation: float | None = None,
    ) -> None:
        cmd = _format_command(pan, tilt, mode, rotation=rotation)
        if self.verbose or self.mock_log:
            print(f"[pico] {cmd.decode().strip()}")
        # Mock mode logs and stops here -- there is no SSH subprocess to
        # write to, only stdout. The link still reports is_connected()
        # so the UI shows "CONNECTED" instead of "OFFLINE".
        if self.mock_log or self._proc is None:
            return
        if self._proc.stdin is None or self._proc.poll() is not None:
            return
        with self._lock:
            try:
                self._proc.stdin.write(cmd)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                print(f"[pico_bridge] write failed: {exc}; bridge dead")
                self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.0)
        except Exception:
            pass
        self._proc = None


# ---------------------------------------------------------------------------
# Calibration UI
# ---------------------------------------------------------------------------

class RemoteCalibrationApp:
    def __init__(
        self,
        mjpeg_url: str,
        ssh_host: str | None,
        cal_path: Path,
        bridge_cmd: str | None,
        ssh_port: int | None,
        mock: bool,
        rotate_180_local: bool,
    ) -> None:
        self.cal_path = cal_path
        self.cal = Calibration.load(cal_path)
        # We don't know the live frame size until the first MJPEG
        # frame arrives; stamp it then.
        self.rotate_180_local = rotate_180_local

        self.source = MJPEGSource(mjpeg_url)
        self.source.start()

        self.link = PicoBridgeLink(
            ssh_host=ssh_host,
            bridge_cmd=bridge_cmd,
            ssh_port=ssh_port,
            mock_log=mock,
            verbose=False,
        )

        self.pan = self.cal.pan_center
        self.tilt = self.cal.tilt_center
        self.rotation = self.cal.rotation_center
        # Park at center on launch. Mode IDLE = no laser pulse.
        self.link.aim(self.pan, self.tilt, PicoMode.IDLE, rotation=self.rotation)

        self.mode = "JOG"
        self.show_help = True
        self.mouse_xy: tuple[int, int] = (
            self.cal.frame_width // 2,
            self.cal.frame_height // 2,
        )
        self.message: tuple[str, float] = ("", 0.0)
        self.window = "BlackFiber Remote Calibration  (h help, q quit)"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._connected_pixel_size_seen = False

        # ---- TRACK-cursor state ---------------------------------------
        # When TRACK is on, the laser is continuously commanded to
        # wherever the mouse cursor sits. Use this to verify the
        # calibration in real time: drag the cursor across the live
        # feed and watch whether the dot follows.
        self._track = False
        self._last_track_xy: tuple[int, int] = (-1, -1)
        self._last_track_send_at = 0.0
        self._track_min_dt = 1.0 / 30.0  # 30 Hz max command rate
        # Cached leave-one-out residuals: anchor index -> (delta_pan,
        # delta_tilt, total_deg). Recomputed when anchors change.
        self._loo: dict[int, tuple[float, float, float]] = {}
        self._loo_dirty = True

    def _flash(self, msg: str) -> None:
        self.message = (msg, time.time() + 2.5)

    def _command_servos(self, mode: PicoMode = PicoMode.IDLE) -> None:
        self.link.aim(self.pan, self.tilt, mode, rotation=self.rotation)

    def _recompute_loo(self) -> None:
        """Leave-one-out residual for every pan/tilt anchor.

        For each anchor i, build a calibration that excludes it, then
        ask that calibration where the laser SHOULD aim to hit anchor
        i's pixel. Compare against anchor i's actual servo angles.
        High residuals = inconsistent anchor (likely a misclick or a
        spot the IDW interpolator can't agree on with its neighbours).

        We display the residual on the anchor's label so the operator
        can spot bad anchors at a glance and remove them with `z`.
        """
        self._loo = {}
        anchors = self.cal.anchors
        if len(anchors) < 3:
            self._loo_dirty = False
            return
        for i, a in enumerate(anchors):
            sub = Calibration(
                anchors=[other for j, other in enumerate(anchors) if j != i],
                rotation_anchors=list(self.cal.rotation_anchors),
                frame_width=self.cal.frame_width,
                frame_height=self.cal.frame_height,
                pan_min=self.cal.pan_min, pan_max=self.cal.pan_max,
                tilt_min=self.cal.tilt_min, tilt_max=self.cal.tilt_max,
                pan_center=self.cal.pan_center, tilt_center=self.cal.tilt_center,
                rotation_min=self.cal.rotation_min, rotation_max=self.cal.rotation_max,
                rotation_center=self.cal.rotation_center,
            )
            try:
                pred_pan, pred_tilt = sub.pixel_to_servo(a.pixel_x, a.pixel_y)
            except Exception:
                continue
            dp, dt = a.pan_angle - pred_pan, a.tilt_angle - pred_tilt
            self._loo[i] = (dp, dt, (dp * dp + dt * dt) ** 0.5)
        self._loo_dirty = False

    def _toggle_track(self) -> None:
        self._track = not self._track
        if self._track:
            if len(self.cal.anchors) < 2:
                self._track = False
                self._flash("Need >= 2 anchors before TRACK can aim")
                return
            self._flash("TRACK ON  -- laser follows cursor (t to stop)")
        else:
            self._flash("TRACK OFF")
            # Snap back to center when leaving track so the gimbal
            # parks at a known pose instead of wherever the cursor
            # last was.
            self.pan = self.cal.pan_center
            self.tilt = self.cal.tilt_center
            self.rotation = self.cal.rotation_center
            self._command_servos(PicoMode.IDLE)

    def _maybe_track(self) -> None:
        if not self._track or len(self.cal.anchors) < 2:
            return
        mx, my = self.mouse_xy
        now = time.perf_counter()
        if (mx, my) == self._last_track_xy and (now - self._last_track_send_at) < 0.25:
            return
        if (now - self._last_track_send_at) < self._track_min_dt:
            return
        pan, tilt = self.cal.pixel_to_servo(float(mx), float(my))
        rotation = self.cal.pixel_to_rotation(float(mx), float(my))
        self.pan, self.tilt, self.rotation = pan, tilt, rotation
        self._command_servos(PicoMode.TRACKING)
        self._last_track_xy = (mx, my)
        self._last_track_send_at = now

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        self.mouse_xy = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if self.mode == "JOG":
            if shift:
                self.cal.add_rotation(
                    RotationAnchor(float(x), float(y), self.rotation)
                )
                self._flash(
                    f"+ rotation anchor ({x},{y}) -> rot={self.rotation:.1f}"
                )
            else:
                self.cal.add(
                    CalibrationAnchor(float(x), float(y), self.pan, self.tilt)
                )
                self._loo_dirty = True
                self._flash(
                    f"+ anchor ({x},{y}) -> pan={self.pan:.1f} tilt={self.tilt:.1f}"
                )
        else:
            if len(self.cal.anchors) < 2:
                self._flash("Need >= 2 pan/tilt anchors before TEST mode aiming")
                return
            pan, tilt = self.cal.pixel_to_servo(float(x), float(y))
            rotation = self.cal.pixel_to_rotation(float(x), float(y))
            self.pan, self.tilt, self.rotation = pan, tilt, rotation
            self._command_servos(PicoMode.TRACKING)
            self._flash(
                f"AIM ({x},{y}) -> pan={pan:.1f} tilt={tilt:.1f} rot={rotation:+.1f}"
            )

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False

        # We treat capslock / SHIFT (uppercase ASCII letters) as the
        # "5x faster jog" modifier, same as calibration_tool.py.
        upper = key in (ord("W"), ord("A"), ord("S"), ord("D"), ord("J"), ord("L"))
        step = JOG_STEP_DEG_SHIFT if upper else JOG_STEP_DEG
        rot_step = JOG_ROT_STEP_DEG_SHIFT if upper else JOG_ROT_STEP_DEG
        pan, tilt, rotation = self.pan, self.tilt, self.rotation

        if key in (ord("w"), ord("W")):
            tilt -= step
        elif key in (ord("s"), ord("S")):
            tilt += step
        elif key in (ord("a"), ord("A")):
            pan -= step
        elif key in (ord("d"), ord("D")):
            pan += step
        elif key in (ord("j"), ord("J")):
            rotation -= rot_step
        elif key in (ord("l"), ord("L")):
            rotation += rot_step
        elif key == ord("r"):
            pan, tilt = self.cal.pan_center, self.cal.tilt_center
            rotation = self.cal.rotation_center
            self._flash("Reset to center")
        elif key == ord("m"):
            # Toggle JOG/TEST. Always exit TRACK first so we don't have
            # the laser auto-aiming while the operator is in JOG mode.
            if self._track:
                self._toggle_track()
            self.mode = "TEST" if self.mode == "JOG" else "JOG"
            self._flash(f"Mode: {self.mode}")
        elif key == ord("t"):
            self._toggle_track()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("f"):
            self.cal.save(self.cal_path)
            self._flash(
                f"Saved {len(self.cal.anchors)}+{len(self.cal.rotation_anchors)}rot "
                f"anchors -> {self.cal_path.name}"
            )
        elif key == ord("g"):
            self.cal = Calibration.load(self.cal_path)
            self._loo_dirty = True
            self._flash(
                f"Loaded {len(self.cal.anchors)}+{len(self.cal.rotation_anchors)}rot "
                f"anchors"
            )
        elif key == ord("c"):
            self.cal.anchors.clear()
            self._loo_dirty = True
            self._flash("Cleared all pan/tilt anchors")
        elif key == ord("C"):
            self.cal.rotation_anchors.clear()
            self._flash("Cleared all rotation anchors")
        elif key == ord("z"):
            mx, my = self.mouse_xy
            removed = self.cal.remove_nearest(float(mx), float(my))
            if removed:
                self._loo_dirty = True
            self._flash(
                "Removed nearest pan/tilt anchor"
                if removed else "No anchor near cursor"
            )
        elif key == ord("Z"):
            mx, my = self.mouse_xy
            removed = self.cal.remove_nearest_rotation(float(mx), float(my))
            self._flash(
                "Removed nearest rotation anchor"
                if removed else "No anchor near cursor"
            )
        elif ord("1") <= key <= ord("9"):
            preset = int(chr(key))
            grid = [(60, 60), (90, 60), (120, 60), (60, 90), (90, 90), (120, 90),
                    (60, 120), (90, 120), (120, 120)]
            pan, tilt = grid[preset - 1]
            self._flash(f"Preset {preset}: pan={pan} tilt={tilt}")

        pan = max(self.cal.pan_min, min(self.cal.pan_max, pan))
        tilt = max(self.cal.tilt_min, min(self.cal.tilt_max, tilt))
        rotation = max(self.cal.rotation_min, min(self.cal.rotation_max, rotation))
        if pan != self.pan or tilt != self.tilt or rotation != self.rotation:
            self.pan, self.tilt, self.rotation = pan, tilt, rotation
            self._command_servos(PicoMode.IDLE)
        return True

    def _draw_overlays(self, frame: np.ndarray) -> None:
        if self._loo_dirty:
            self._recompute_loo()

        for i, a in enumerate(self.cal.anchors):
            # Color-code by leave-one-out residual:
            #   GREEN  < 2 deg  (anchor agrees with neighbours)
            #   AMBER  2-5 deg  (mild disagreement, still usable)
            #   RED    > 5 deg  (probably misclicked / inconsistent)
            #   CYAN   no LOO yet (need >= 3 anchors)
            err = self._loo.get(i)
            if err is None:
                color = (220, 220, 0)  # cyan-ish: not yet evaluated
                label_extra = ""
            else:
                _dp, _dt, mag = err
                if mag < 2.0:
                    color = (0, 220, 0)
                elif mag < 5.0:
                    color = (0, 200, 240)
                else:
                    color = (0, 0, 230)
                label_extra = f" e={mag:.1f}d"
            cv2.circle(frame, (int(a.pixel_x), int(a.pixel_y)), 8,
                       color, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(a.pixel_x), int(a.pixel_y)), 2,
                       color, -1, cv2.LINE_AA)
            _shadowed_text(
                frame,
                f"{i + 1} ({a.pan_angle:.0f},{a.tilt_angle:.0f}){label_extra}",
                (int(a.pixel_x) + 12, int(a.pixel_y) - 8),
                scale=0.4,
                thickness=1,
            )

        for i, a in enumerate(self.cal.rotation_anchors):
            cv2.circle(frame, (int(a.pixel_x), int(a.pixel_y)), 10,
                       (200, 100, 255), 2, cv2.LINE_AA)
            _shadowed_text(
                frame,
                f"R{i + 1} {a.rotation_angle:+.0f}",
                (int(a.pixel_x) + 12, int(a.pixel_y) + 14),
                scale=0.4,
                thickness=1,
            )

        # Predicted-aim crosshair: TEST mode shows it on hover; TRACK
        # mode renders a bigger one so the operator can see the laser
        # target prominently while the dot follows in the live feed.
        if (self.mode == "TEST" or self._track) and len(self.cal.anchors) >= 2:
            mx, my = self.mouse_xy
            pred_pan, pred_tilt = self.cal.pixel_to_servo(float(mx), float(my))
            pred_rot = self.cal.pixel_to_rotation(float(mx), float(my))
            marker_color = (0, 230, 120) if self._track else (0, 0, 255)
            marker_size = 28 if self._track else 18
            cv2.drawMarker(frame, (mx, my), marker_color,
                           cv2.MARKER_CROSS, marker_size, 2)
            cv2.circle(frame, (mx, my), 14, marker_color, 1, cv2.LINE_AA)
            _shadowed_text(
                frame,
                f"aim: pan={pred_pan:.1f} tilt={pred_tilt:.1f} rot={pred_rot:+.1f}",
                (mx + 16, my + 18),
                scale=0.45,
                thickness=1,
            )

        draw_crosshair(frame)

        h, w = frame.shape[:2]
        bridge_state = "CONNECTED" if self.link.is_connected() else "OFFLINE"
        src_err = self.source.last_error()
        feed_state = "OFFLINE: " + src_err if src_err else "LIVE"
        # Aggregate LOO summary: how many anchors are in each tier.
        loo_counts = [0, 0, 0]  # green, amber, red
        for _i, (_dp, _dt, mag) in self._loo.items():
            if mag < 2.0:
                loo_counts[0] += 1
            elif mag < 5.0:
                loo_counts[1] += 1
            else:
                loo_counts[2] += 1
        loo_str = (
            f"FIT: {loo_counts[0]}g / {loo_counts[1]}a / {loo_counts[2]}r"
            if self._loo else "FIT: -- (need >=3 anchors)"
        )
        mode_disp = "TRACK" if self._track else self.mode
        bar_lines = [
            f"MODE: {mode_disp}   PAN: {self.pan:6.1f}   TILT: {self.tilt:6.1f}   "
            f"ROT: {self.rotation:+6.1f}",
            f"ANCHORS: {len(self.cal.anchors)} pan/tilt + "
            f"{len(self.cal.rotation_anchors)} rot   "
            f"{loo_str}   "
            f"PICO: {bridge_state}   FEED: {feed_state}   "
            f"FILE: {self.cal_path.name}",
        ]
        y = h - 16 - 18 * (len(bar_lines) - 1)
        for line in bar_lines:
            _shadowed_text(frame, line, (12, y), scale=0.55, thickness=1)
            y += 18

        if self.show_help:
            if self._track:
                second_line = "TRACK ON: laser follows cursor live (t to stop)"
            elif self.mode == "JOG":
                second_line = "LEFT-CLICK: add pan/tilt anchor    SHIFT+CLICK: add rotation anchor"
            else:
                second_line = "LEFT-CLICK: AIM at clicked pixel via calibration"
            help_lines = [
                "WASD: nudge pan/tilt 1deg     J/L: nudge stepper 2deg    (capslock = 5x)",
                second_line,
                "M: switch JOG/TEST    T: live TRACK cursor    R: reset to center    1-9: presets",
                "Z / shift-Z: remove nearest pan-tilt / rotation anchor    C / shift-C: clear all",
                "F: save    G: load    H: toggle help    Q/Esc: quit",
                "Anchor color = leave-one-out residual: green<2deg  amber<5deg  red>5deg",
            ]
            box_h = 22 * len(help_lines) + 12
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 70), (700, 70 + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            for i, line in enumerate(help_lines):
                _shadowed_text(frame, line, (20, 92 + i * 22), scale=0.5, thickness=1)

        if self.message[1] > time.time():
            _shadowed_text(frame, self.message[0], (12, 50),
                           scale=0.6, color=(0, 255, 255), thickness=2)

    def _waiting_frame(self) -> np.ndarray:
        # Render a placeholder so the user sees SOMETHING while we wait
        # for the first MJPEG frame to arrive.
        canvas = np.zeros((480, 854, 3), dtype=np.uint8)
        msg = "Waiting for MJPEG stream..."
        err = self.source.last_error()
        sub = err if err else "is jetson_live_detect.py running with --mjpeg?"
        _shadowed_text(canvas, msg, (40, 220), scale=1.0, thickness=2)
        _shadowed_text(canvas, sub, (40, 260), scale=0.55,
                       color=(0, 200, 255), thickness=1)
        return canvas

    def run(self) -> int:
        try:
            while True:
                ok, frame, _seq = self.source.read()
                if not ok or frame is None:
                    frame = self._waiting_frame()
                else:
                    if self.rotate_180_local:
                        # Convenience: rotate on the Mac side instead of the
                        # Jetson. Use this if the Jetson is streaming a
                        # non-rotated feed and you don't want to restart it.
                        # For production calibration, prefer rotating on the
                        # Jetson so the bytes hitting the disk match what
                        # the runtime sees.
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if not self._connected_pixel_size_seen:
                        h, w = frame.shape[:2]
                        self.cal.frame_width = w
                        self.cal.frame_height = h
                        self.mouse_xy = (w // 2, h // 2)
                        self._connected_pixel_size_seen = True
                self._maybe_track()
                self._draw_overlays(frame)
                cv2.imshow(self.window, frame)
                key = cv2.waitKey(15) & 0xFF
                if key != 255:
                    if not self._handle_key(key):
                        break
        finally:
            self.source.stop()
            self.link.close()
            cv2.destroyAllWindows()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mjpeg",
        default="http://localhost:8765/stream.mjpg",
        help="MJPEG stream URL. Default assumes you've SSH-tunneled "
             "the Jetson's port 8765 to the Mac (ssh -fN -L 8765:127.0.0.1:8765 cask). "
             "Use http://<jetson-ip>:8765/stream.mjpg to skip the tunnel.",
    )
    parser.add_argument(
        "--ssh", default="cask",
        help="SSH host alias for the Jetson (default 'cask'). The script "
             "spawns one persistent SSH connection that runs the Pico bridge.",
    )
    parser.add_argument(
        "--ssh-port", type=int, default=None,
        help="Optional SSH port if you don't have it in ~/.ssh/config.",
    )
    parser.add_argument(
        "--bridge-cmd",
        default=None,
        help="Override the remote bridge command (default: "
             "'python3 ~/Katena/scripts/pico_bridge.py --no-telemetry'). "
             "Use this if your Jetson user / paths differ.",
    )
    parser.add_argument(
        "--cal", type=Path, default=PROJECT_ROOT / "calibration.json",
        help="Path to the local calibration JSON (default ./calibration.json). "
             "Loaded on start, saved with F. After you save, scp this file to "
             "the Jetson before running jetson_live_detect with --pico-port.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Don't open SSH at all; log every Pico command to stdout. "
             "Pairs well with a recorded MJPEG (or any HTTP MJPEG you have) "
             "for verifying the UI without touching hardware.",
    )
    parser.add_argument(
        "--rotate-180", action="store_true",
        help="Rotate frames 180deg on the Mac before showing them. ONLY use "
             "this for one-off testing -- in production, rotate on the Jetson "
             "(jetson_live_detect.py --rotate-180) so the calibration anchors "
             "are saved against bytes identical to what the runtime sees.",
    )
    args = parser.parse_args()

    if args.mock:
        print("[mock] no SSH connection; commands will be printed to stdout")
    elif args.ssh:
        if args.bridge_cmd:
            shown = args.bridge_cmd
        else:
            shown = "python3 ~/Katena/scripts/pico_bridge.py --no-telemetry"
        print(f"[ssh] {args.ssh}: {shlex.quote(shown)}")

    app = RemoteCalibrationApp(
        mjpeg_url=args.mjpeg,
        ssh_host=None if args.mock else args.ssh,
        cal_path=args.cal,
        bridge_cmd=args.bridge_cmd,
        ssh_port=args.ssh_port,
        mock=args.mock,
        rotate_180_local=args.rotate_180,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
