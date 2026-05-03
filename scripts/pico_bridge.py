"""Minimal stdin -> serial bridge for the Pico.

This script runs on the Jetson (the machine the Pico is physically
plugged into) and is driven over SSH by tools running on the Mac --
primarily `scripts/calibration_remote.py`. It does ONE thing: read
command lines from stdin and write them to `/dev/ttyACM0` (or whatever
`--port` you give it). All telemetry coming back from the Pico is
streamed to stdout as raw lines so the caller can tail it.

Why a separate process instead of just `ssh cask 'echo ... > /dev/ttyACM0'`:
  - One persistent SSH session = one TTY open per session = no
    "device or resource busy" thrash.
  - Single Python process owns the file handle, so flow control and
    RX-buffer drain stay sane (the Pico happily streams telemetry
    every 100 ms and would back-pressure a naive write).
  - Bidirectional: callers can read telemetry off our stdout if they
    want; ignore it if they don't.

Protocol (matches the Pico firmware in `pico/pico_controller.py`):
  Host -> Pico:  "P{pan}T{tilt}[R{rot}]M{mode}\n"   (3-127 ASCII bytes)
  Pico -> Host:  "D{cm}S{stat}L{ldr}A{rot}P{pir}\n"  (telemetry @ ~10 Hz)

Typical usage from a parent process:

    proc = subprocess.Popen(
        ["ssh", "cask", "python3 ~/Katena/scripts/pico_bridge.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    proc.stdin.write(b"P95T80M0\n")
    proc.stdin.flush()

CLI:
    python3 scripts/pico_bridge.py
    python3 scripts/pico_bridge.py --port /dev/ttyACM0 --baud 115200
    python3 scripts/pico_bridge.py --no-telemetry   # don't echo Pico lines

The script terminates on stdin EOF (parent closed the pipe) or SIGINT.
"""

from __future__ import annotations

import argparse
import select
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Serial port to the Pico (default /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="Baudrate (default 115200; the Pico CDC ignores this anyway)")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="Discard Pico->Host telemetry instead of echoing it on stdout. "
                         "Use this when the parent process doesn't care, to avoid "
                         "unnecessary work on the SSH pipe.")
    ap.add_argument("--idle-print-s", type=float, default=5.0,
                    help="Print a 'pico_bridge: alive' heartbeat every N seconds on "
                         "stderr so the parent can see we're still up. 0 = never.")
    args = ap.parse_args()

    try:
        # write_timeout: don't let a backed-up Pico CDC block us forever.
        # If a write times out we just drop that command -- callers will
        # send fresh state on the next tick anyway.
        ser = serial.Serial(
            args.port, args.baud, timeout=0, write_timeout=0.2
        )
    except serial.SerialException as exc:
        sys.stderr.write(f"pico_bridge: cannot open {args.port}: {exc}\n")
        return 1

    sys.stderr.write(f"pico_bridge: opened {args.port} @ {args.baud} baud\n")
    sys.stderr.flush()

    last_alive = time.monotonic()
    stdin_fd = sys.stdin.fileno()
    serial_fd = ser.fileno()

    # Use select() for both stdin and the serial port so we never busy-wait
    # and never block on one stream while data is waiting on the other.
    try:
        while True:
            rlist, _, _ = select.select([stdin_fd, serial_fd], [], [], 0.5)

            if stdin_fd in rlist:
                line = sys.stdin.buffer.readline()
                if not line:
                    sys.stderr.write("pico_bridge: stdin EOF, exiting\n")
                    break
                # Make sure every command ends with a newline, the Pico's
                # firmware-side parser is line-oriented and silently drops
                # incomplete lines.
                if not line.endswith(b"\n"):
                    line += b"\n"
                try:
                    ser.write(line)
                except serial.SerialTimeoutException:
                    sys.stderr.write("pico_bridge: serial write timeout (dropped)\n")
                except serial.SerialException as exc:
                    sys.stderr.write(f"pico_bridge: serial write failed: {exc}\n")
                    break

            if serial_fd in rlist:
                try:
                    chunk = ser.read(256)
                except serial.SerialException:
                    chunk = b""
                if chunk and not args.no_telemetry:
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    except (BrokenPipeError, OSError):
                        sys.stderr.write("pico_bridge: stdout closed, exiting\n")
                        break

            now = time.monotonic()
            if args.idle_print_s > 0 and (now - last_alive) >= args.idle_print_s:
                sys.stderr.write(f"pico_bridge: alive t={now:.0f}s\n")
                sys.stderr.flush()
                last_alive = now
    except KeyboardInterrupt:
        sys.stderr.write("pico_bridge: SIGINT, exiting\n")
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
