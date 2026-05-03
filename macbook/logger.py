"""Engagement logger abstraction.

Two implementations behind the same interface:
  - LocalJSONLogger:    always-on, writes JSONL to disk
  - FoundryOSDKLogger:  syncs to Foundry via OSDK (stubbed until OSDK
                        package is generated from Developer Console)
  - DualLogger:         writes to both, so the local file is always a
                        complete record even if Foundry sync drops

The local log is the system of record on the edge. Foundry is the
common operating picture. They are reconciled by drone_id.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .engagement import DroneEngagement


class EngagementLogger(ABC):
    @abstractmethod
    def create(self, eng: DroneEngagement) -> None: ...

    @abstractmethod
    def update(self, eng: DroneEngagement) -> None: ...

    @abstractmethod
    def list_recent(self, limit: int = 20) -> list[DroneEngagement]: ...

    def close(self) -> None:
        return None


class LocalJSONLogger(EngagementLogger):
    """Append-only JSONL file. One line per record (create or update).

    Record format: { "op": "create"|"update", "ts": iso8601, "eng": {...} }
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, op: str, eng: DroneEngagement) -> None:
        record = {"op": op, "eng": eng.to_dict()}
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()

    def create(self, eng: DroneEngagement) -> None:
        self._append("create", eng)

    def update(self, eng: DroneEngagement) -> None:
        self._append("update", eng)

    def list_recent(self, limit: int = 20) -> list[DroneEngagement]:
        if not self.path.exists():
            return []
        latest_by_id: dict[str, dict[str, Any]] = {}
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eng = rec["eng"]
                    latest_by_id[eng["drone_id"]] = eng
                except (json.JSONDecodeError, KeyError):
                    continue
        ordered = sorted(
            latest_by_id.values(),
            key=lambda e: e.get("detection_timestamp", ""),
            reverse=True,
        )
        return [DroneEngagement.from_dict(e) for e in ordered[:limit]]


class FoundryOSDKLogger(EngagementLogger):
    """Stub. Fill in once the OSDK client is generated from Developer Console.

    Expected wiring (rough sketch — actual API names depend on what
    Developer Console emits for your specific ontology):

        from osdk_drone_neutralizer import FoundryClient, DroneEngagement as FE
        self.client = FoundryClient(...)

        def create(self, eng):
            self.client.objects.DroneEngagement.create(
                drone_id=eng.drone_id,
                detection_timestamp=eng.detection_timestamp,
                ...
            )

        def update(self, eng):
            self.client.objects.DroneEngagement(eng.drone_id).update(
                fiber_cut=eng.fiber_cut, ...
            )
    """

    def __init__(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def create(self, eng: DroneEngagement) -> None:
        return None

    def update(self, eng: DroneEngagement) -> None:
        return None

    def list_recent(self, limit: int = 20) -> list[DroneEngagement]:
        return []


class DualLogger(EngagementLogger):
    """Write to both loggers. Local always succeeds; Foundry failures are swallowed."""

    def __init__(self, primary: EngagementLogger, secondary: EngagementLogger) -> None:
        self.primary = primary
        self.secondary = secondary

    def create(self, eng: DroneEngagement) -> None:
        self.primary.create(eng)
        try:
            self.secondary.create(eng)
        except Exception:
            pass

    def update(self, eng: DroneEngagement) -> None:
        self.primary.update(eng)
        try:
            self.secondary.update(eng)
        except Exception:
            pass

    def list_recent(self, limit: int = 20) -> list[DroneEngagement]:
        return self.primary.list_recent(limit)


_counter_lock = threading.Lock()
_counter = 0


def next_drone_id(prefix: str = "FOG") -> str:
    """Sequential, dramatic-looking IDs: FOG-001, FOG-002, ..."""
    global _counter
    with _counter_lock:
        _counter += 1
        return f"{prefix}-{_counter:03d}"
