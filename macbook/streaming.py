"""Tiny MJPEG-over-HTTP streamer for the live tracking pipeline.

Why not the snapshot/rsync trick:
    Polling a JPEG file at 0.5s intervals gives 2 fps and a noticeable
    lag. For testing the pipeline live (e.g. flying the drone in front of
    the Jetson camera and checking that the detector follows it) we need
    real video framerate. MJPEG (multipart/x-mixed-replace) is the
    simplest possible way to get that into a browser tab on the Mac with
    no extra client software.

Usage from a producer (camera loop):

    from macbook.streaming import MJPEGServer

    streamer = MJPEGServer(port=8765, jpeg_quality=85)
    streamer.start()  # spins up daemon HTTP thread

    while True:
        frame = capture()
        annotate(frame)
        streamer.publish(frame)   # encodes to JPEG once and shares it

    # then on the Mac:
    #   open  http://<jetson-host>:8765/   in a browser tab

Threading model:
    One background HTTP server thread accepts connections; each viewer
    spawns its own request thread (ThreadingHTTPServer). They all read
    the same `_latest_jpeg` bytes guarded by a short lock + Condition,
    so a slow viewer can't block the producer.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time
from typing import Optional

import cv2
import numpy as np


_INDEX_HTML = """<!doctype html>
<html><head><title>BlackFiber live</title>
<style>
  body { background:#0b0d10; color:#e5e7eb; font:14px ui-monospace,monospace;
         margin:0; padding:16px; }
  h1 { font-size:14px; letter-spacing:0.05em; color:#9ca3af; margin:0 0 12px;
       display:flex; align-items:center; gap:14px; }
  img { display:block; max-width:100%; height:auto;
        border:1px solid #1f2937; border-radius:6px; box-shadow:0 0 24px #00000088; }
  button { font:inherit; padding:6px 14px; border-radius:4px; border:1px solid #4b5563;
           background:#1f2937; color:#e5e7eb; cursor:pointer; letter-spacing:0.04em; }
  button:hover { background:#374151; }
  button.rec   { background:#7f1d1d; border-color:#b91c1c; }
  button.rec:hover { background:#991b1b; }
  button.eng   { background:#14532d; border-color:#16a34a; }
  button.eng:hover { background:#166534; }
  #status, #lstatus { color:#6b7280; font-size:12px; }
  #status.on { color:#fca5a5; }
  #lstatus.on { color:#86efac; }
</style></head>
<body>
  <h1>BLACKFIBER LIVE  /stream.mjpg
    <button id="engBtn" onclick="toggleEng()">ENGAGE LASER</button>
    <span id="lstatus">parked</span>
    <span style="opacity:0.4">|</span>
    <button id="recBtn" onclick="toggleRec()">REC</button>
    <span id="status">idle</span>
  </h1>
  <img src="/stream.mjpg" alt="live stream">
<script>
  async function refresh(){
    const r = await fetch('/record/status'); const j = await r.json();
    const btn = document.getElementById('recBtn');
    const sts = document.getElementById('status');
    if (j.recording) { btn.classList.add('rec'); btn.textContent='STOP';
                       sts.classList.add('on'); sts.textContent = 'rec -> '+j.path; }
    else             { btn.classList.remove('rec'); btn.textContent='REC';
                       sts.classList.remove('on'); sts.textContent =
                         j.last_path ? 'last: '+j.last_path : 'idle'; }
    const lr = await fetch('/laser/status'); const lj = await lr.json();
    const lbtn = document.getElementById('engBtn');
    const lsts = document.getElementById('lstatus');
    if (lj.engaged) { lbtn.classList.add('eng'); lbtn.textContent='DISENGAGE';
                      lsts.classList.add('on'); lsts.textContent = 'tracking live'; }
    else            { lbtn.classList.remove('eng'); lbtn.textContent='ENGAGE LASER';
                      lsts.classList.remove('on'); lsts.textContent = 'parked'; }
  }
  async function toggleRec(){
    await fetch('/record/toggle', {method:'POST'}); refresh();
  }
  async function toggleEng(){
    await fetch('/laser/toggle', {method:'POST'}); refresh();
  }
  refresh(); setInterval(refresh, 800);
</script>
</body></html>
"""


class MJPEGServer:
    """Daemon HTTP server that streams the latest frame as MJPEG."""

    def __init__(
        self,
        port: int = 8765,
        host: str = "0.0.0.0",
        jpeg_quality: int = 85,
        max_fps: float = 30.0,
    ) -> None:
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be 1..100")
        if max_fps <= 0:
            raise ValueError("max_fps must be > 0")
        self.port = port
        self.host = host
        self.jpeg_quality = jpeg_quality
        self.min_interval_s = 1.0 / max_fps

        self._latest_jpeg: Optional[bytes] = None
        self._latest_seq = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._server: Optional[socketserver.BaseServer] = None
        self._thread: Optional[threading.Thread] = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        # --- recording-control state (browser-driven). The producer
        # loop polls these to decide whether to open / close a writer.
        # We don't own the cv2.VideoWriter here because it must run on
        # the producer thread that has the frame dims and codec setup.
        self._record_lock = threading.Lock()
        self._record_requested = False
        self._record_active_path: Optional[str] = None  # set by producer
        self._record_last_path: Optional[str] = None    # set by producer at stop

        # --- laser-engage state (browser-driven). Producer polls
        # `laser_engaged` each frame to decide whether to actually
        # actuate the gimbal. Defaults to FALSE so the laser never
        # starts moving on its own at process launch -- the operator
        # has to click ENGAGE LASER on the browser page first.
        self._laser_lock = threading.Lock()
        self._laser_engaged = False

    # ----- producer side -----

    def publish(self, frame_bgr: np.ndarray) -> bool:
        """Encode the frame to JPEG and notify any viewers. Returns True
        on success, False if encoding failed."""
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, self._encode_params)
        if not ok:
            return False
        data = jpeg.tobytes()
        with self._cond:
            self._latest_jpeg = data
            self._latest_seq += 1
            self._cond.notify_all()
        return True

    # ----- recording control (thread-safe) -----

    @property
    def recording_requested(self) -> bool:
        """Producer polls this each frame to decide writer state."""
        with self._record_lock:
            return self._record_requested

    def set_recording_active(self, path: Optional[str]) -> None:
        """Producer calls this after it (re)opens or closes the writer.

        Pass the path string when recording has started, or None when
        the writer has been closed. The HTTP /record/status endpoint
        reads from this so the browser shows the actual file path.
        """
        with self._record_lock:
            if path is None and self._record_active_path is not None:
                self._record_last_path = self._record_active_path
            self._record_active_path = path

    def _toggle_record(self) -> dict:
        with self._record_lock:
            self._record_requested = not self._record_requested
            return {
                "recording": self._record_requested,
                "path": self._record_active_path,
                "last_path": self._record_last_path,
            }

    def _record_status(self) -> dict:
        with self._record_lock:
            return {
                "recording": self._record_requested,
                "path": self._record_active_path,
                "last_path": self._record_last_path,
            }

    # ----- laser-engage control (thread-safe) -----

    @property
    def laser_engaged(self) -> bool:
        """Producer polls this each frame. While False the producer
        should keep the gimbal parked at calibration center and skip
        all FIRE actuation -- still rendering the overlays so the
        operator can see what WOULD happen, just not moving servos."""
        with self._laser_lock:
            return self._laser_engaged

    def set_laser_engaged(self, engaged: bool) -> None:
        """Force the engage state from code (e.g. CLI --engage-on-start
        or a safety auto-disengage). Browser button uses _toggle_laser."""
        with self._laser_lock:
            self._laser_engaged = bool(engaged)

    def _toggle_laser(self) -> dict:
        with self._laser_lock:
            self._laser_engaged = not self._laser_engaged
            return {"engaged": self._laser_engaged}

    def _laser_status(self) -> dict:
        with self._laser_lock:
            return {"engaged": self._laser_engaged}

    # ----- server lifecycle -----

    def start(self) -> int:
        srv = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            # Silence per-request access logs -- noisy at 30 fps.
            def log_message(self, *args, **kwargs) -> None:  # noqa: ARG002
                return

            def _serve_index(self) -> None:
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self) -> None:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=--frameboundary",
                )
                self.end_headers()
                last_seq = -1
                last_send_at = 0.0
                while True:
                    with srv._cond:
                        # Wait for a NEW frame (sequence number bumped).
                        if srv._latest_jpeg is None or srv._latest_seq == last_seq:
                            srv._cond.wait(timeout=2.0)
                        data = srv._latest_jpeg
                        seq = srv._latest_seq
                    if data is None:
                        # No frames yet; small sleep, keep connection alive.
                        time.sleep(0.05)
                        continue
                    if seq == last_seq:
                        continue
                    now = time.perf_counter()
                    wait = srv.min_interval_s - (now - last_send_at)
                    if wait > 0:
                        time.sleep(wait)
                    last_seq = seq
                    last_send_at = time.perf_counter()
                    try:
                        self.wfile.write(b"--frameboundary\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return  # viewer disconnected; tear down handler

            def _serve_json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
                if self.path in ("/", "/index.html"):
                    self._serve_index()
                elif self.path in ("/stream.mjpg", "/stream"):
                    self._serve_stream()
                elif self.path == "/record/status":
                    self._serve_json(srv._record_status())
                elif self.path == "/laser/status":
                    self._serve_json(srv._laser_status())
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/record/toggle":
                    self._serve_json(srv._toggle_record())
                elif self.path == "/laser/toggle":
                    self._serve_json(srv._toggle_laser())
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _ThreadingServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"mjpeg-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None
