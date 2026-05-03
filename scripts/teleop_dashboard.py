"""Live dashboard for the Pico teleop.

Reads the JSONL log produced by `scripts/teleop_pico.py` and renders a
small live status panel in your terminal. Two ways to source the log:

    Local file (teleop running on this machine):
        python scripts/teleop_dashboard.py --log ~/Katena/logs/teleop_pico.log

    Remote file via SSH (typical: teleop on Jetson, dashboard on Mac):
        python scripts/teleop_dashboard.py --ssh cask
        python scripts/teleop_dashboard.py --ssh cask \\
               --remote-log ~/Katena/logs/teleop_pico.log

Renders a panel like:

    +------------------------------------------------------------+
    | Katena Pico teleop                       cask:teleop_pico  |
    |                                                            |
    |  pan  :  87.5 deg   ->  target  87.5 deg                   |
    |  tilt :  92.0 deg   ->  target  92.0 deg                   |
    |                                                            |
    |  mode : 1 (TRACKING)        step: 1.0 deg                  |
    |  rate : 20.0 deg/s          last update: 0.04s ago         |
    +------------------------------------------------------------+

Press Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODE_NAMES = {0: "IDLE", 1: "TRACKING", 2: "SWEEP", 3: "LOCKED"}


def _open_remote_tail(host: str, remote_path: str) -> subprocess.Popen:
    cmd = [
        "ssh",
        "-o", "ServerAliveInterval=15",
        "-o", "ConnectTimeout=5",
        host,
        f"mkdir -p {remote_path!s}/.. 2>/dev/null; touch {remote_path}; tail -n 1 -F {remote_path}",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def _open_local_tail(local_path: Path) -> subprocess.Popen:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.touch(exist_ok=True)
    return subprocess.Popen(
        ["tail", "-n", "1", "-F", str(local_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def _bar(value: float, lo: float = 0.0, hi: float = 180.0, width: int = 30) -> str:
    """Tiny ASCII bar showing where 'value' sits between lo and hi."""
    if hi <= lo:
        return "[" + " " * width + "]"
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    pos = int(round(frac * (width - 1)))
    inside = ["-"] * width
    middle = (width - 1) // 2
    inside[middle] = "|"
    inside[pos] = "#"
    return "[" + "".join(inside) + "]"


def _render(state: dict, source_label: str, last_update_age: float) -> str:
    pan_pos = state.get("pan_pos", 0.0)
    tilt_pos = state.get("tilt_pos", 0.0)
    pan_tgt = state.get("pan_target", pan_pos)
    tilt_tgt = state.get("tilt_target", tilt_pos)
    step = state.get("step", 0.0)
    rate = state.get("max_rate", 0.0)
    mode = state.get("mode", 0)

    width = max(64, min(shutil.get_terminal_size((80, 20)).columns - 2, 92))
    border = "+" + "-" * (width - 2) + "+"

    def line(s: str) -> str:
        return "| " + s.ljust(width - 4) + " |"

    title = f"Katena Pico teleop"
    pad = " " * max(1, width - 4 - len(title) - len(source_label))
    header = title + pad + source_label

    age_str = f"{last_update_age:0.2f}s ago" if last_update_age < 9.99 else f"{int(last_update_age)}s ago"
    age_warn = " (stale!)" if last_update_age > 2.0 else ""

    rows = [
        border,
        line(header),
        line(""),
        line(f"pan   : {pan_pos:6.1f} deg   ->  target {pan_tgt:6.1f} deg"),
        line(f"        {_bar(pan_pos)}"),
        line(""),
        line(f"tilt  : {tilt_pos:6.1f} deg   ->  target {tilt_tgt:6.1f} deg"),
        line(f"        {_bar(tilt_pos)}"),
        line(""),
        line(f"mode  : {mode} ({MODE_NAMES.get(mode, '?'):<8})    step: {step:.1f} deg"),
        line(f"rate  : {rate:5.1f} deg/s        last update: {age_str}{age_warn}"),
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
    src.add_argument("--log", type=Path,
                     help="Local JSONL log file produced by teleop_pico.py")
    src.add_argument("--ssh", metavar="HOST",
                     help="SSH host where the log lives (e.g. 'cask')")
    ap.add_argument("--remote-log", default="~/Katena/logs/teleop_pico.log",
                    help="Remote log path when --ssh is used "
                         "(default ~/Katena/logs/teleop_pico.log)")
    ap.add_argument("--no-clear", action="store_true",
                    help="Don't clear the screen between renders "
                         "(useful for piping or no-tty environments)")
    args = ap.parse_args()

    if args.ssh:
        proc = _open_remote_tail(args.ssh, args.remote_log)
        source_label = f"{args.ssh}:{args.remote_log.split('/')[-1]}"
    else:
        local = args.log or (Path.home() / "Katena/logs/teleop_pico.log")
        proc = _open_local_tail(local)
        source_label = f"local:{local.name}"

    state: dict = {}
    last_update = time.monotonic()
    last_render = 0.0

    try:
        assert proc.stdout is not None
        # Non-blocking-ish reads: poll for new lines every 100ms,
        # re-render even if no new data so the "stale!" age updates.
        import select
        while True:
            r, _, _ = select.select([proc.stdout], [], [], 0.1)
            if r:
                line = proc.stdout.readline()
                if not line:
                    # tail process died (eg. ssh disconnect)
                    err = proc.stderr.read() if proc.stderr else ""
                    sys.stderr.write(f"\n[tail process exited]\n{err}\n")
                    return 1
                line = line.strip()
                if line:
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        evt = None
                    if evt is not None:
                        # start/stop events don't carry pan/tilt, so we
                        # only refresh the displayed state when we have
                        # actual servo numbers. Otherwise the panel
                        # would blank out to zeros at shutdown.
                        if "pan_pos" in evt:
                            state = evt
                        last_update = time.monotonic()

            now = time.monotonic()
            if now - last_render >= 0.1:
                if not args.no_clear:
                    _clear_screen()
                age = now - last_update
                if state:
                    sys.stdout.write(_render(state, source_label, age) + "\n")
                else:
                    sys.stdout.write(
                        f"Waiting for first event from {source_label}...\n"
                    )
                sys.stdout.flush()
                last_render = now
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
