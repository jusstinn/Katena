"""Seed engagements.jsonl with realistic-looking historical data.

Two purposes:
  1) Verify the logger -> JSONL -> dashboard pipeline end-to-end without
     needing a camera or hardware.
  2) Pre-mint a 'Threat Library' so the live demo dashboard already
     shows recent ops history when judges arrive. The next live engagement
     becomes "Engagement #N+1 today" rather than "Engagement #1".

Usage:
    python scripts/seed_engagements.py                 # 6 engagements over last 2h
    python scripts/seed_engagements.py --count 12      # more
    python scripts/seed_engagements.py --reset         # truncate file first
"""

import argparse
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from macbook.engagement import DroneEngagement, ThreatLevel  # noqa: E402
from macbook.logger import LocalJSONLogger, next_drone_id  # noqa: E402

OUTCOMES = [
    {"weight": 7, "fiber_cut": True,  "signal_lost": True,  "notes": ""},
    {"weight": 1, "fiber_cut": False, "signal_lost": True,  "notes": "lost"},
    {"weight": 1, "fiber_cut": False, "signal_lost": False, "notes": "stand down"},
    {"weight": 1, "fiber_cut": True,  "signal_lost": True,  "notes": "swarm engagement"},
]

THREATS = [
    (ThreatLevel.HIGH,     5),
    (ThreatLevel.CRITICAL, 3),
    (ThreatLevel.MEDIUM,   2),
]


def _weighted_choice(items: list[dict]) -> dict:
    weights = [it["weight"] for it in items]
    return random.choices(items, weights=weights, k=1)[0]


def _weighted_threat() -> ThreatLevel:
    levels, weights = zip(*THREATS)
    return random.choices(levels, weights=weights, k=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "engagements.jsonl")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--span-min", type=int, default=120, help="Spread engagements over the last N minutes")
    parser.add_argument("--reset", action="store_true", help="Truncate the file before seeding")
    args = parser.parse_args()

    if args.reset and args.log.exists():
        args.log.unlink()
        print(f"Truncated {args.log}")

    logger = LocalJSONLogger(args.log)
    now = datetime.now(timezone.utc)
    rng = random.Random(time.time_ns() & 0xFFFFFFFF)

    print(f"Seeding {args.count} engagements into {args.log} ...")
    for i in range(args.count):
        minutes_ago = (args.span_min / max(args.count, 1)) * (args.count - i - 1) + rng.uniform(0, 5)
        ts = now - timedelta(minutes=minutes_ago)

        outcome = _weighted_choice(OUTCOMES)
        threat = _weighted_threat()
        sensors = ["camera"]
        if rng.random() < 0.7:
            sensors.append("rf_silence")
        if rng.random() < 0.3:
            sensors.append("ultrasonic")

        eng = DroneEngagement(
            drone_id=next_drone_id(),
            detection_timestamp=ts.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            position_x=round(rng.uniform(100, 1100), 1),
            position_y=round(rng.uniform(80, 600), 1),
            position_z=round(rng.uniform(8.0, 30.0), 1),
            threat_level=threat,
            sweep_radius_m=round(rng.uniform(1.5, 3.5), 1),
            sensor_fusion=sensors,
            rf_silence_confirmed="rf_silence" in sensors,
            pan_angle=round(rng.uniform(60, 120), 1),
            tilt_angle=round(rng.uniform(60, 120), 1),
            notes=outcome["notes"],
        )
        logger.create(eng)

        if outcome["fiber_cut"] or outcome["signal_lost"]:
            duration = round(rng.uniform(4.0, 14.0), 2)
            eng.engagement_start = ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            eng.engagement_end = (ts + timedelta(seconds=duration)).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            eng.engagement_duration_s = duration
            eng.fiber_cut = outcome["fiber_cut"]
            eng.signal_lost = outcome["signal_lost"]
            eng.signal_strength = 0.0 if outcome["fiber_cut"] else round(rng.uniform(0.4, 0.8), 2)
        logger.update(eng)

        marker = "X" if outcome["fiber_cut"] else "·"
        print(
            f"  [{marker}] {eng.drone_id}  {ts.strftime('%H:%M:%S')}  "
            f"{threat.value:8s}  sensors={','.join(sensors):20s}  "
            f"dur={(eng.engagement_duration_s or 0):5.2f}s  notes={eng.notes!r}"
        )

    print(f"\nDone. {args.count} engagements written.")
    print(f"View with:  streamlit run dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
