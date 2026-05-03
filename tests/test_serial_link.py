"""Tests for the Pico W serial bridge — formatting, parsing, mock simulation."""

from __future__ import annotations

import time

import pytest

from macbook.serial_link import (
    MockPicoLink,
    PicoMode,
    _format_command,
    _parse_telemetry,
    open_link,
)


class TestFormatCommand:
    def test_round_trip_through_pico_parser(self, pico_controller):
        for pan, tilt, mode in [
            (95.5, 47.2, PicoMode.SWEEP),
            (0.0, 180.0, PicoMode.IDLE),
            (90.0, 90.0, PicoMode.LOCKED),
            (1.0, 179.0, PicoMode.TRACKING),
        ]:
            cmd = _format_command(pan, tilt, mode).decode().rstrip()
            parsed = pico_controller._parse_command(cmd)
            assert parsed is not None, cmd
            p, t, r, m = parsed
            assert abs(p - pan) < 0.1
            assert abs(t - tilt) < 0.1
            assert r is None
            assert m == int(mode)

    def test_round_trip_with_rotation(self, pico_controller):
        for pan, tilt, rot, mode in [
            (95.5, 47.2, 12.5, PicoMode.SWEEP),
            (0.0, 180.0, -90.0, PicoMode.IDLE),
            (90.0, 90.0, 0.0, PicoMode.LOCKED),
            (90.0, 90.0, 179.9, PicoMode.TRACKING),
        ]:
            cmd = _format_command(pan, tilt, mode, rotation=rot).decode().rstrip()
            parsed = pico_controller._parse_command(cmd)
            assert parsed is not None, cmd
            p, t, r, m = parsed
            assert abs(p - pan) < 0.1
            assert abs(t - tilt) < 0.1
            assert abs(r - rot) < 0.1
            assert m == int(mode)

    def test_rotation_clamped_to_180(self):
        cmd = _format_command(90, 90, PicoMode.IDLE, rotation=999)
        assert b"R180.0" in cmd
        cmd = _format_command(90, 90, PicoMode.IDLE, rotation=-999)
        assert b"R-180.0" in cmd

    def test_clamps_out_of_range(self):
        cmd = _format_command(-50, 999, PicoMode.IDLE)
        assert b"P0.0" in cmd
        assert b"T180.0" in cmd

    def test_command_ends_in_newline(self):
        cmd = _format_command(90, 90, PicoMode.IDLE)
        assert cmd.endswith(b"\n")
        cmd = _format_command(90, 90, PicoMode.IDLE, rotation=45.0)
        assert cmd.endswith(b"\n")

    def test_no_rotation_omits_R_field(self):
        cmd = _format_command(90, 90, PicoMode.IDLE)
        assert b"R" not in cmd

    @pytest.mark.parametrize("mode", list(PicoMode))
    def test_all_modes_serialize(self, mode):
        cmd = _format_command(90, 90, mode)
        assert f"M{int(mode)}".encode() in cmd


class TestParseTelemetry:
    def test_basic_line(self):
        tel = _parse_telemetry("D45.7S2L450")
        assert tel is not None
        assert tel.distance_cm == 45.7
        assert tel.status == 2
        assert tel.ldr_raw == 450

    def test_full_signal(self):
        tel = _parse_telemetry("D100S0L900", ldr_full=900, ldr_cut=50)
        assert tel.fiber_signal == pytest.approx(1.0)

    def test_cut_signal(self):
        tel = _parse_telemetry("D100S3L50", ldr_full=900, ldr_cut=50)
        assert tel.fiber_signal == pytest.approx(0.0)

    def test_below_cut_clamps_to_zero(self):
        tel = _parse_telemetry("D100S3L0", ldr_full=900, ldr_cut=50)
        assert tel.fiber_signal == pytest.approx(0.0)

    def test_invalid_distance(self):
        tel = _parse_telemetry("D-1.0S0L500")
        assert tel.distance_cm is None

    def test_garbage_line_returns_none(self):
        for line in ["", "garbage", "DSL", "P90T90M0"]:
            assert _parse_telemetry(line) is None

    def test_parses_rotation_when_present(self):
        tel = _parse_telemetry("D45.7S2L450A37.5")
        assert tel is not None
        assert tel.rotation_deg == 37.5

    def test_parses_negative_rotation(self):
        tel = _parse_telemetry("D45.7S2L450A-90.0")
        assert tel.rotation_deg == -90.0

    def test_legacy_telemetry_without_rotation_still_parses(self):
        tel = _parse_telemetry("D45.7S2L450")
        assert tel is not None
        assert tel.rotation_deg is None


class TestMockPicoLink:
    def test_always_connected(self):
        link = MockPicoLink()
        assert link.is_connected() is True

    def test_idle_keeps_signal_full(self):
        link = MockPicoLink()
        for _ in range(5):
            assert link.telemetry().fiber_signal == pytest.approx(1.0)

    def test_locked_drives_signal_to_zero(self):
        link = MockPicoLink(severance_seconds=0.5)
        link.aim(90, 90, PicoMode.LOCKED)
        time.sleep(0.6)
        tel = link.telemetry()
        assert tel.fiber_signal == 0.0

    def test_severance_is_monotonic(self):
        link = MockPicoLink(severance_seconds=1.0)
        link.aim(90, 90, PicoMode.LOCKED)
        readings = []
        for _ in range(8):
            readings.append(link.telemetry().fiber_signal)
            time.sleep(0.1)
        for a, b in zip(readings, readings[1:]):
            assert b <= a + 0.05

    def test_severance_floor_persists_after_disengage(self):
        """Severance is only committed to the floor when telemetry is polled
        while LOCKED — that's how the real tracker uses it (every frame).
        Once committed, the floor only recovers very slowly (0.001/poll)."""
        link = MockPicoLink(severance_seconds=0.3)
        link.aim(90, 90, PicoMode.LOCKED)
        for _ in range(6):
            time.sleep(0.07)
            link.telemetry()  # commit severance to floor
        link.aim(90, 90, PicoMode.IDLE)
        assert link.telemetry().fiber_signal < 0.05

    def test_reset_severance_restores_signal(self):
        link = MockPicoLink(severance_seconds=0.2)
        link.aim(90, 90, PicoMode.LOCKED)
        time.sleep(0.3)
        link.reset_severance()
        link.aim(90, 90, PicoMode.IDLE)
        assert link.telemetry().fiber_signal == pytest.approx(1.0, abs=0.01)

    def test_distance_within_realistic_range(self):
        link = MockPicoLink(mock_distance_cm=200.0)
        for _ in range(5):
            tel = link.telemetry()
            assert 195.0 <= tel.distance_cm <= 205.0

    def test_aim_records_last_position(self):
        link = MockPicoLink()
        link.aim(45.5, 67.2, PicoMode.TRACKING)
        assert link._last_pan == 45.5
        assert link._last_tilt == 67.2

    def test_rotation_target_advances_toward_value(self):
        link = MockPicoLink(rotation_speed_deg_s=1000.0)
        link.aim(90, 90, PicoMode.TRACKING, rotation=45.0)
        time.sleep(0.1)
        tel = link.telemetry()
        assert tel.rotation_deg == pytest.approx(45.0, abs=1.0)

    def test_rotation_omitted_keeps_previous_target(self):
        link = MockPicoLink(rotation_speed_deg_s=1000.0)
        link.aim(90, 90, PicoMode.TRACKING, rotation=30.0)
        time.sleep(0.1)
        link.telemetry()
        link.aim(91, 91, PicoMode.TRACKING)
        time.sleep(0.1)
        tel = link.telemetry()
        assert tel.rotation_deg == pytest.approx(30.0, abs=1.0)

    def test_rotation_clamped_in_aim(self):
        link = MockPicoLink(rotation_speed_deg_s=1000.0)
        link.aim(90, 90, PicoMode.TRACKING, rotation=999.0)
        assert link._rotation_target == 180.0
        link.aim(90, 90, PicoMode.TRACKING, rotation=-999.0)
        assert link._rotation_target == -180.0

    def test_telemetry_includes_rotation_field(self):
        link = MockPicoLink()
        tel = link.telemetry()
        assert tel.rotation_deg is not None
        assert tel.rotation_deg == 0.0


class TestOpenLinkFactory:
    def test_mock_flag_returns_mock(self):
        link = open_link(mock=True)
        assert isinstance(link, MockPicoLink)

    def test_falls_back_to_mock_when_port_missing(self):
        link = open_link(mock=False, port="/dev/cu.this-port-does-not-exist-katena")
        assert isinstance(link, MockPicoLink)
