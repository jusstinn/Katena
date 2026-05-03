"""Slow / safe WASD teleop for the Pico's pan/tilt servos.

Pin map on the Pico (matches `pico/pico_controller.py`):
    GP0  = pan  servo (X axis)
    GP1  = tilt servo (Y axis)

Wire protocol the firmware accepts (115200 baud over USB CDC ACM):
    "P{pan}T{tilt}M{mode}\\n"   pan/tilt 0-180 deg, mode 0-3

Safety design (to avoid breaking servos / cables):

  * NO command is sent on launch -- the servos stay exactly where they
    physically are. The script just *believes* the Pico is at the
    --start-pan / --start-tilt angles (default 90/90, matching the
    firmware's power-on state).
  * WASD does NOT issue a discrete jump command. It only nudges a
    *target* angle. A 50 Hz interpolator then walks the actually-sent
    angle towards that target at most --max-rate degrees per second
    (default 20 deg/s -- visibly slow). Each frame moves at most
    `max_rate / 50 ~= 0.4 deg`, so even held-down keys produce smooth
    motion that the servos can track without straining.
  * `z` re-zeroes the script's internal belief to the configured
    start angles WITHOUT moving the servo -- use this if you've
    manually repositioned the rig and want WASD to start from there.
  * `c` (center) gracefully ramps to (90, 90) at the same rate limit;
    it does NOT slam.

Keys:
    a / d         pan  - / +    (slow ramp)
    w / s         tilt + / -    (slow ramp; w = up, s = down)
    z             zero: declare "current position = start angles"
                  (script belief only, no motion)
    c             center: smoothly ramp to (90, 90)
    [ / ]         step size - / +    (deg added to target per keypress)
    , / .         rate limit - / +   (max deg/sec the interpolator uses)
    1 .. 3        Pico mode (1=tracking 2=sweep 3=locked, 0=idle)
    h             show help
    q / Esc       quit (servos hold last commanded position)

Run on the Jetson (Pico is /dev/ttyACM0 there):

    cd ~/Katena
    ./.venv-jetson/bin/python scripts/teleop_pico.py

If you've manually positioned the servos somewhere other than
(90, 90), tell the script so the first move is a tiny nudge from
the right belief:

    python scripts/teleop_pico.py --start-pan 70 --start-tilt 110

Mock mode (no hardware, just prints what would be sent):

    python scripts/teleop_pico.py --mock
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path


@dataclass
class State:
    pan_target: float
    tilt_target: float
    pan_pos: float                              # the angle we last actually sent
    tilt_pos: float
    step: float                                 # deg added to target per keypress
    max_rate: float                             # deg / second
    mode: int = 1                               # 1 = TRACKING by default
    last_sent: tuple[float, float, int] | None = None


HELP_TEXT = (
    "  a/d : pan -/+    w/s : tilt +/-    z : zero    c : center    "
    "[/] : step    ,/. : rate    1-3 : mode    h : help    q : quit"
)

EPS_DEG = 0.05         # don't bother sending a frame if pos changed less than this


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _format_command(pan: float, tilt: float, mode: int) -> bytes:
    pan = _clamp(pan, 0.0, 180.0)
    tilt = _clamp(tilt, 0.0, 180.0)
    return f"P{pan:.1f}T{tilt:.1f}M{int(mode)}\n".encode()


class _SerialOut:
    def __init__(self, port: str | None, baud: int, mock: bool) -> None:
        self.mock = mock
        self.ser = None
        if mock:
            return
        import serial
        # Disable HW flow control explicitly. Default for pyserial is
        # already off, but being explicit avoids surprises across
        # versions / OSes.
        self.ser = serial.Serial(
            port, baud,
            timeout=0,
            write_timeout=0.05,
            rtscts=False, dsrdtr=False, xonxoff=False,
        )

    def write(self, payload: bytes) -> None:
        if self.mock:
            sys.stdout.write(f"\n[mock] -> {payload.decode().rstrip()}\n")
            sys.stdout.flush()
            return
        assert self.ser is not None
        try:
            self.ser.write(payload)
        except Exception:
            # write_timeout exceeded -- USB CDC backpressure.
            # Best to just drop this command than hang the main loop;
            # next tick will resend an updated position anyway.
            pass

    def drain_rx(self) -> None:
        """Drop any unread bytes from the device.

        We never read telemetry in this script, so without this the
        kernel TTY RX buffer fills, USB CDC flow-control engages, and
        our writes start blocking. Calling this once per tick keeps
        the buffer near-empty.
        """
        if self.ser is None:
            return
        try:
            n = self.ser.in_waiting
            if n > 0:
                self.ser.read(n)
        except Exception:
            pass

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def _draw_status(state: State, conn_label: str) -> None:
    # Show both pos (what we've sent) and target (where you've nudged it
    # to) so you can see motion catching up.
    line = (
        f"\rpos=({state.pan_pos:6.1f},{state.tilt_pos:6.1f})  "
        f"tgt=({state.pan_target:6.1f},{state.tilt_target:6.1f})  "
        f"step={state.step:4.1f}  rate={state.max_rate:5.1f}d/s  "
        f"mode={state.mode}  ({conn_label})   "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def _read_key_nonblocking(timeout_s: float) -> str | None:
    r, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.005)
        if not r:
            break
        sys.stdin.read(1)
    return "\x1b"


def teleop(args: argparse.Namespace) -> int:
    state = State(
        pan_target=args.start_pan,
        tilt_target=args.start_tilt,
        pan_pos=args.start_pan,
        tilt_pos=args.start_tilt,
        step=args.step,
        max_rate=args.max_rate,
    )
    # Prime last_sent to match our belief so the interpolator stays
    # silent until the user actually nudges something. Without this
    # the first tick would emit a command -- harmless if our belief
    # matches the firmware's, but a SLAM if the user used
    # --start-pan/--start-tilt to assert a different physical pose.
    state.last_sent = (state.pan_pos, state.tilt_pos, state.mode)

    out = _SerialOut(port=args.port, baud=args.baud, mock=args.mock)
    conn_label = "MOCK" if args.mock else f"port={args.port}"

    # Optional JSONL state log so a dashboard / tailer can mirror what
    # the servos are doing in real time. Empty path disables logging.
    log_path: Path | None = None
    log_fp = None
    if args.log:
        log_path = Path(args.log).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", buffering=1)  # line-buffered
        log_fp.write(
            json.dumps({
                "t": time.time(),
                "event": "start",
                "conn": conn_label,
                "start_pan": args.start_pan,
                "start_tilt": args.start_tilt,
                "max_rate": args.max_rate,
                "step": args.step,
            }) + "\n"
        )

    def _log(event: str) -> None:
        if log_fp is None:
            return
        log_fp.write(
            json.dumps({
                "t": time.time(),
                "event": event,
                "pan_pos": round(state.pan_pos, 2),
                "tilt_pos": round(state.tilt_pos, 2),
                "pan_target": round(state.pan_target, 2),
                "tilt_target": round(state.tilt_target, 2),
                "step": state.step,
                "max_rate": state.max_rate,
                "mode": state.mode,
            }) + "\n"
        )

    def _send_if_changed() -> None:
        cmd = (state.pan_pos, state.tilt_pos, state.mode)
        if cmd == state.last_sent:
            return
        out.write(_format_command(*cmd))
        state.last_sent = cmd
        _log("send")

    sys.stdout.write(
        "Pico teleop ready. NO command sent yet -- servos stay where they are.\n"
        f"Belief: pan={state.pan_pos:.1f} tilt={state.tilt_pos:.1f} "
        f"(use --start-pan/--start-tilt or 'z' to update if wrong).\n"
    )
    sys.stdout.write(HELP_TEXT + "\n")

    fd = sys.stdin.fileno()
    is_tty = sys.stdin.isatty()
    old_attrs = termios.tcgetattr(fd) if is_tty else None
    if is_tty:
        tty.setcbreak(fd)

    def _restore(*_a) -> None:
        if old_attrs is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass
        out.close()
        if log_fp is not None:
            try:
                log_fp.write(json.dumps({"t": time.time(), "event": "stop"}) + "\n")
                log_fp.close()
            except Exception:
                pass
        sys.stdout.write("\n")
        sys.stdout.flush()

    signal.signal(signal.SIGINT, lambda *_a: (_restore(), os._exit(130)))
    signal.signal(signal.SIGTERM, lambda *_a: (_restore(), os._exit(143)))

    tick_dt = 1.0 / 50.0                       # 50 Hz interpolator
    last_tick = time.monotonic()
    last_status_at = 0.0
    last_heartbeat_at = 0.0
    heartbeat_interval = 0.5                   # log even when idle, every 500ms

    try:
        while True:
            # Block at most one tick worth of time waiting for a key.
            key = _read_key_nonblocking(timeout_s=tick_dt)

            if key is not None:
                k = key.lower()
                if k in ("q", "\x1b"):
                    break
                elif k == "a":
                    state.pan_target = _clamp(state.pan_target - state.step, 0.0, 180.0)
                elif k == "d":
                    state.pan_target = _clamp(state.pan_target + state.step, 0.0, 180.0)
                elif k == "w":
                    state.tilt_target = _clamp(state.tilt_target + state.step, 0.0, 180.0)
                elif k == "s":
                    state.tilt_target = _clamp(state.tilt_target - state.step, 0.0, 180.0)
                elif k == "z":
                    # Re-declare current physical pose to be the configured
                    # start angles. No motion -- pos AND target snap to the
                    # same value, and we mark last_sent matching so the
                    # interpolator stays quiet.
                    state.pan_pos = args.start_pan
                    state.tilt_pos = args.start_tilt
                    state.pan_target = args.start_pan
                    state.tilt_target = args.start_tilt
                    state.last_sent = (state.pan_pos, state.tilt_pos, state.mode)
                elif k == "c":
                    state.pan_target = 90.0
                    state.tilt_target = 90.0
                elif k == "[":
                    state.step = max(0.5, state.step - 0.5)
                elif k == "]":
                    state.step = min(20.0, state.step + 0.5)
                elif k == ",":
                    state.max_rate = max(2.0, state.max_rate - 2.0)
                elif k == ".":
                    state.max_rate = min(120.0, state.max_rate + 2.0)
                elif k in ("0", "1", "2", "3"):
                    state.mode = int(k)
                    # Force a re-send so the firmware picks up the new mode.
                    state.last_sent = None
                elif k == "h":
                    sys.stdout.write("\n" + HELP_TEXT + "\n")
                    sys.stdout.flush()

            now = time.monotonic()
            dt = now - last_tick
            if dt >= tick_dt:
                last_tick = now
                # Drop any unread telemetry from the Pico so the kernel
                # TTY RX buffer doesn't fill up and back-pressure our
                # writes via USB CDC flow control. (Without this, after
                # ~2 minutes the script's main loop hangs for many
                # seconds at a time waiting for write() to unblock.)
                out.drain_rx()
                # Walk pos toward target at max_rate deg/sec.
                max_step = state.max_rate * dt
                for axis in ("pan", "tilt"):
                    pos = getattr(state, f"{axis}_pos")
                    tgt = getattr(state, f"{axis}_target")
                    delta = tgt - pos
                    if abs(delta) <= max_step:
                        new_pos = tgt
                    else:
                        new_pos = pos + (max_step if delta > 0 else -max_step)
                    setattr(state, f"{axis}_pos", new_pos)
                # Only send if pos actually changed enough to matter, OR
                # if last_sent is None (mode change / never-sent).
                if (
                    state.last_sent is None
                    or abs(state.pan_pos - state.last_sent[0]) >= EPS_DEG
                    or abs(state.tilt_pos - state.last_sent[1]) >= EPS_DEG
                    or state.mode != state.last_sent[2]
                ):
                    _send_if_changed()

                if now - last_status_at > 0.05:
                    _draw_status(state, conn_label)
                    last_status_at = now
                if now - last_heartbeat_at > heartbeat_interval:
                    _log("heartbeat")
                    last_heartbeat_at = now
    finally:
        _restore()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Serial device for the Pico (default /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--start-pan", type=float, default=90.0,
                    help="Pan angle the Pico is currently believed to be at "
                         "(default 90, matching firmware power-on). The "
                         "first WASD press will nudge from this value.")
    ap.add_argument("--start-tilt", type=float, default=90.0,
                    help="Tilt angle the Pico is currently believed to be "
                         "at (default 90).")
    ap.add_argument("--step", type=float, default=1.0,
                    help="Degrees added to the TARGET per keypress "
                         "(default 1.0 -- single tap = ~1 deg of motion).")
    ap.add_argument("--max-rate", type=float, default=20.0,
                    help="Max servo speed the interpolator will use, "
                         "deg/sec (default 20 -- intentionally slow).")
    ap.add_argument("--mock", action="store_true",
                    help="Don't open the serial port; print commands instead")
    ap.add_argument("--log", default=str(Path.home() / "Katena/logs/teleop_pico.log"),
                    help="JSONL state log (one line per command + heartbeat). "
                         "Empty string disables logging.")
    args = ap.parse_args()
    return teleop(args)


if __name__ == "__main__":
    sys.exit(main())
