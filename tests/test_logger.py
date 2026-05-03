"""Tests for the engagement logger abstraction."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from macbook.engagement import DroneEngagement, ThreatLevel
from macbook.logger import (
    DualLogger,
    FoundryOSDKLogger,
    LocalJSONLogger,
    next_drone_id,
)


class TestNextDroneId:
    def test_ids_are_sequential(self):
        a = next_drone_id()
        b = next_drone_id()
        assert a != b
        assert a.startswith("FOG-")
        assert b.startswith("FOG-")

    def test_custom_prefix(self):
        d = next_drone_id(prefix="TEST")
        assert d.startswith("TEST-")


class TestLocalJSONLogger:
    def test_create_writes_one_line(self, tmp_jsonl: Path):
        log = LocalJSONLogger(tmp_jsonl)
        log.create(DroneEngagement(drone_id="X"))
        assert tmp_jsonl.exists()
        lines = tmp_jsonl.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["op"] == "create"

    def test_update_appends_separate_record(self, tmp_jsonl: Path):
        log = LocalJSONLogger(tmp_jsonl)
        e = DroneEngagement(drone_id="X")
        log.create(e)
        e.fiber_cut = True
        log.update(e)
        lines = tmp_jsonl.read_text().strip().splitlines()
        assert len(lines) == 2
        ops = [json.loads(line)["op"] for line in lines]
        assert ops == ["create", "update"]

    def test_list_recent_returns_latest_per_id(self, tmp_jsonl: Path):
        log = LocalJSONLogger(tmp_jsonl)
        a = DroneEngagement(drone_id="A")
        b = DroneEngagement(drone_id="B")
        log.create(a); log.create(b)
        a.notes = "v2"
        log.update(a)
        recent = log.list_recent(limit=10)
        assert len(recent) == 2
        a_record = next(r for r in recent if r.drone_id == "A")
        assert a_record.notes == "v2"

    def test_list_recent_handles_missing_file(self, tmp_path: Path):
        log = LocalJSONLogger(tmp_path / "nope.jsonl")
        assert log.list_recent() == []

    def test_list_recent_skips_malformed_lines(self, tmp_jsonl: Path):
        tmp_jsonl.write_text(
            'not-json\n'
            '{"op":"create","eng":{"drone_id":"OK","sensor_fusion":["camera"]}}\n'
            '{"missing_eng":true}\n'
        )
        log = LocalJSONLogger(tmp_jsonl)
        recent = log.list_recent()
        assert len(recent) == 1
        assert recent[0].drone_id == "OK"

    def test_concurrent_writes_are_safe(self, tmp_jsonl: Path):
        log = LocalJSONLogger(tmp_jsonl)
        N = 50

        def writer(i: int) -> None:
            log.create(DroneEngagement(drone_id=f"T-{i:03d}"))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = tmp_jsonl.read_text().strip().splitlines()
        assert len(lines) == N
        for line in lines:
            json.loads(line)


class TestFoundryOSDKLoggerStub:
    def test_stub_silently_succeeds(self):
        log = FoundryOSDKLogger()
        e = DroneEngagement(drone_id="X")
        log.create(e)
        log.update(e)
        assert log.list_recent() == []
        assert log.is_connected() is False


class TestDualLogger:
    def test_writes_to_both(self, tmp_jsonl: Path):
        primary = LocalJSONLogger(tmp_jsonl)

        secondary_calls: list[str] = []

        class Tap(FoundryOSDKLogger):
            def create(self, eng):
                secondary_calls.append(f"create:{eng.drone_id}")

            def update(self, eng):
                secondary_calls.append(f"update:{eng.drone_id}")

        dual = DualLogger(primary, Tap())
        e = DroneEngagement(drone_id="X")
        dual.create(e)
        dual.update(e)

        assert len(tmp_jsonl.read_text().strip().splitlines()) == 2
        assert secondary_calls == ["create:X", "update:X"]

    def test_secondary_failure_does_not_break_primary(self, tmp_jsonl: Path):
        primary = LocalJSONLogger(tmp_jsonl)

        class Broken(FoundryOSDKLogger):
            def create(self, eng):
                raise RuntimeError("network down")

            def update(self, eng):
                raise RuntimeError("network down")

        dual = DualLogger(primary, Broken())
        e = DroneEngagement(drone_id="X")
        dual.create(e)
        dual.update(e)
        assert tmp_jsonl.read_text().count("\n") == 2

    def test_list_recent_reads_from_primary(self, tmp_jsonl: Path):
        primary = LocalJSONLogger(tmp_jsonl)
        dual = DualLogger(primary, FoundryOSDKLogger())
        dual.create(DroneEngagement(drone_id="X"))
        assert len(dual.list_recent()) == 1
