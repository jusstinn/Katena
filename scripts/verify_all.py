"""Run every verification script and print a consolidated summary.

This is the script to run before each demo. If everything is green,
you're cleared for engagement.

Usage:
    python scripts/verify_all.py
"""

import subprocess
import sys
from pathlib import Path

CHECKS = [
    ("camera", ["python", "scripts/verify_camera.py"]),
    ("yolo", ["python", "scripts/verify_yolo.py"]),
    ("serial", ["python", "scripts/verify_serial.py"]),
    ("foundry", ["python", "scripts/verify_foundry.py"]),
]


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    results: list[tuple[str, bool, str]] = []
    for name, cmd in CHECKS:
        print(f"\n=== {name.upper()} =========================================")
        try:
            proc = subprocess.run(
                cmd,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            results.append((name, False, "timeout"))
            print("TIMEOUT")
            continue
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        ok = proc.returncode == 0
        last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        results.append((name, ok, last_line))

    print("\n=== SUMMARY ============================================")
    width = max(len(n) for n, _, _ in results)
    for name, ok, last in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {name.ljust(width)}  {last}")
    failures = sum(1 for _, ok, _ in results if not ok)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
