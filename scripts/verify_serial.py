"""Serial connectivity smoke test.

Lists all available serial ports. With --connect, opens the configured
PICO_SERIAL_PORT and tries to read a line. Useful to confirm the Pico W
is mounted as a USB serial device after flashing MicroPython.

Usage:
    python scripts/verify_serial.py                # list ports
    python scripts/verify_serial.py --connect      # try to read from PICO_SERIAL_PORT
"""

import argparse
import os
import sys
import time

import serial
import serial.tools.list_ports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true", help="Try opening configured port")
    parser.add_argument("--port", default=None, help="Override port path")
    parser.add_argument("--baud", type=int, default=int(os.getenv("PICO_BAUDRATE", "115200")))
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    print("Available serial ports:")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  (none)")
    for p in ports:
        print(f"  {p.device}    {p.description}")
        if p.vid is not None:
            print(f"      VID:PID = {p.vid:04x}:{p.pid:04x}")

    if not args.connect:
        return 0

    target = args.port or os.getenv("PICO_SERIAL_PORT", "/dev/cu.usbmodem101")
    print(f"\nAttempting to open {target} @ {args.baud} baud (timeout {args.timeout}s)...")
    try:
        with serial.Serial(target, args.baud, timeout=args.timeout) as s:
            print("OK: port opened.")
            time.sleep(0.5)
            line = s.readline()
            if line:
                print(f"  Read: {line!r}")
            else:
                print("  (no data received within timeout — Pico may not be sending yet)")
    except serial.SerialException as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
