"""Live dashboard for the PIR / Fresnel proximity sensor on the Pico.

Reads the Pico's telemetry stream (the line-oriented
"D{cm}S{status}L{ldr}A{rot}P{pir}" format printed every 100 ms by
pico/pico_controller.py) and renders a clean, screenshot-friendly
status panel showing whether a bystander is detected in front of the
rig.

This is OBSERVE-ONLY for now. It does not disarm the laser or alter
any FIRE state -- it just visualises what the safety interlock would
see if/when we wire it into the engagement state machine.

Two ways to source the data:

    Local (Pico plugged into THIS machine):
        python scripts/pir_dashboard.py --port /dev/tty.usbmodem*

    Over SSH (Pico on the Jetson, dashboard on Mac):
        python scripts/pir_dashboard.py --ssh cask
        python scripts/pir_dashboard.py --ssh cask --remote-port /dev/ttyACM0

The dashboard polls at 10 Hz, the firmware emits at 10 Hz, so the
displayed PIR state is at most ~200 ms behind reality.

Press Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


# Matches the firmware's printed telemetry, with PIR field optional so
# the dashboard still renders something useful against an old firmware
# that hasn't been re-flashed yet (PIR will just stay "no data").
_TELEM_RE = re.compile(
    r"D(?P<d>-?\d+(?:\.\d+)?)"
    r"S(?P<s>\d+)"
    r"L(?P<l>\d+)"
    r"(?:A(?P<a>-?\d+(?:\.\d+)?))?"
    r"(?:P(?P<p>[01]))?"
)


@dataclass
class Stats:
    pir: int | None = None             # last PIR reading (None = no firmware data yet)
    triggers: int = 0                  # rising-edge count since dashboard started
    last_trigger_at: float | None = None
    last_clear_at: float | None = None
    last_telem_at: float | None = None
    started_at: float = 0.0
    distance_cm: float | None = None
    mode: int | None = None
    ldr: int | None = None


def _open_local(port: str, baud: int) -> subprocess.Popen:
    """Read raw lines from a local serial device using a tiny inline reader."""
    code = (
        "import serial,sys\n"
        f"s=serial.Serial({port!r},{baud},timeout=0.2)\n"
        "buf=b''\n"
        "while True:\n"
        "  buf+=s.read(256)\n"
        "  while b'\\n' in buf:\n"
        "    line,_,buf=buf.partition(b'\\n')\n"
        "    sys.stdout.write(line.decode(errors='replace').rstrip()+'\\n')\n"
        "    sys.stdout.flush()\n"
    )
    return subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def _open_ssh(host: str, remote_port: str, baud: int) -> subprocess.Popen:
    """SSH into HOST and stream the same line-by-line reader from there."""
    py = (
        "python3 -c \"import serial,sys;"
        f"s=serial.Serial('{remote_port}',{baud},timeout=0.2);"
        "buf=b''\n"
        "while True:\n"
        "  buf+=s.read(256)\n"
        "  while b'\\n' in buf:\n"
        "    line,_,buf=buf.partition(b'\\n')\n"
        "    sys.stdout.write(line.decode(errors='replace').rstrip()+chr(10))\n"
        "    sys.stdout.flush()\n"
        "\""
    )
    # Fall back to .venv-jetson python if system python3 lacks pyserial.
    venv_py = "~/Katena/.venv-jetson/bin/python -u"
    fallback = py.replace("python3 -c", f"{venv_py} -c")
    cmd = [
        "ssh",
        "-o", "ServerAliveInterval=15",
        "-o", "ConnectTimeout=5",
        host,
        f"({py}) 2>/dev/null || ({fallback})",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 1.0:
        return f"{int(seconds * 1000)} ms ago"
    if seconds < 60.0:
        return f"{seconds:.1f} s ago"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s ago"


def _fmt_uptime(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s}s"
    h = m // 60
    m = m % 60
    return f"{h}h {m}m"


def _render(stats: Stats, source_label: str, telem_age: float) -> str:
    now_t = time.monotonic()
    last_trigger_age = (
        (now_t - stats.last_trigger_at) if stats.last_trigger_at is not None else None
    )
    width = max(64, min(shutil.get_terminal_size((80, 28)).columns - 2, 88))
    border = "+" + "-" * (width - 2) + "+"

    def line(s: str = "") -> str:
        return "| " + s.ljust(width - 4) + " |"

    def centered(s: str) -> str:
        return "| " + s.center(width - 4) + " |"

    title = "BlackFiber PIR Safety Interlock"
    pad = " " * max(1, width - 4 - len(title) - len(source_label))
    header = title + pad + source_label

    if stats.pir is None:
        big_line = "[ NO TELEMETRY YET ]"
        sep = "-" * (width - 4)
        status_text = "Waiting for first reading from the Pico..."
    elif stats.pir == 1:
        big_line = "!!  BYSTANDER DETECTED  !!"
        sep = "#" * (width - 4)
        status_text = (
            "Interlock would DISARM the laser now (observe-only mode)."
        )
    else:
        big_line = "AREA CLEAR"
        sep = "=" * (width - 4)
        status_text = "Interlock condition: safe to engage."

    if telem_age > 1.5:
        status_text += f"   (stale telemetry: {telem_age:.1f}s old)"

    rows: list[str] = [
        border,
        line(header),
        line(),
        line(sep),
        line(),
        centered(big_line),
        line(),
        line(sep),
        line(),
        line(status_text),
        line(),
        line(f"PIR sensor      : GP4 (digital)"),
        line(f"PIR raw         : {stats.pir if stats.pir is not None else '?'}"),
        line(f"Last trigger    : {_fmt_age(last_trigger_age)}"),
        line(f"Triggers total  : {stats.triggers} since dashboard started"),
        line(f"Pico mode       : {stats.mode if stats.mode is not None else '?'}"),
        line(f"Distance (cm)   : {stats.distance_cm if stats.distance_cm is not None else '-'}"),
        line(f"LDR (raw)       : {stats.ldr if stats.ldr is not None else '-'}"),
        line(f"Telem rate      : ~10 Hz   last update: {_fmt_age(telem_age)}"),
        line(f"Dashboard up    : {_fmt_uptime(time.monotonic() - stats.started_at)}"),
        border,
    ]
    return "\n".join(rows)


def _clear_screen() -> None:
    sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--port", default=None,
                     help="Local serial device (e.g. /dev/tty.usbmodem*) "
                          "if the Pico is plugged into THIS machine.")
    src.add_argument("--ssh", metavar="HOST",
                     help="SSH host where the Pico lives "
                          "(e.g. 'cask' for the Jetson).")
    ap.add_argument("--remote-port", default="/dev/ttyACM0",
                    help="Serial device on the SSH host "
                         "(default /dev/ttyACM0).")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-clear", action="store_true",
                    help="Don't clear the screen between renders "
                         "(useful for piping or non-tty output).")
    args = ap.parse_args()

    if args.ssh:
        proc = _open_ssh(args.ssh, args.remote_port, args.baud)
        source_label = f"{args.ssh}:{args.remote_port}"
    else:
        port = args.port or "/dev/ttyACM0"
        proc = _open_local(port, args.baud)
        source_label = f"local:{port}"

    stats = Stats(started_at=time.monotonic())

    def _shutdown(*_a):
        try:
            proc.terminate()
        except Exception:
            pass
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    last_render = 0.0
    import select
    try:
        assert proc.stdout is not None
        while True:
            r, _, _ = select.select([proc.stdout], [], [], 0.1)
            if r:
                line = proc.stdout.readline()
                if not line:
                    err = proc.stderr.read() if proc.stderr else ""
                    sys.stderr.write(f"\n[reader exited]\n{err}\n")
                    return 1
                m = _TELEM_RE.search(line.strip())
                if m:
                    now_t = time.monotonic()
                    stats.last_telem_at = now_t
                    if m.group("d"):
                        stats.distance_cm = float(m.group("d"))
                    if m.group("s"):
                        stats.mode = int(m.group("s"))
                    if m.group("l"):
                        stats.ldr = int(m.group("l"))
                    if m.group("p") is not None:
                        new_pir = int(m.group("p"))
                        if stats.pir == 0 and new_pir == 1:
                            stats.triggers += 1
                            stats.last_trigger_at = now_t
                        if new_pir == 0 and stats.pir != 0:
                            stats.last_clear_at = now_t
                        stats.pir = new_pir

            now = time.monotonic()
            if now - last_render >= 0.1:
                if not args.no_clear:
                    _clear_screen()
                age = (now - stats.last_telem_at) if stats.last_telem_at else 999.0
                sys.stdout.write(_render(stats, source_label, age) + "\n")
                sys.stdout.flush()
                last_render = now
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
