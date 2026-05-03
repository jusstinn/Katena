"""Tests for the Pico W firmware (running under the `machine` shim).

We exercise the pure-logic functions (parser, servo math) directly.
The hardware interaction layers (Servo, Ultrasonic, StatusOutput) are
exercised through the shim to verify they at least construct and don't
explode.
"""

from __future__ import annotations

import pytest


class TestPicoCommandParser:
    def test_valid_command(self, pico_controller):
        assert pico_controller._parse_command("P90.0T45.5M2") == (90.0, 45.5, 2)

    def test_integer_angles(self, pico_controller):
        assert pico_controller._parse_command("P90T45M0") == (90.0, 45.0, 0)

    def test_zero_and_extreme_angles(self, pico_controller):
        assert pico_controller._parse_command("P0.0T180.0M3") == (0.0, 180.0, 3)

    @pytest.mark.parametrize("bad", [
        "garbage",
        "P90T90",
        "Pxxx",
        "P90T90M5",
        "P90T90M-1",
        "",
        "PTM",
        "P90T90M",
        "P90TM0",
    ])
    def test_invalid_inputs_return_none(self, pico_controller, bad):
        assert pico_controller._parse_command(bad) is None

    def test_trailing_whitespace_tolerated(self, pico_controller):
        assert pico_controller._parse_command("P90T90M0  ") == (90.0, 90.0, 0)

    def test_negative_angles_pass_through(self, pico_controller):
        # Parser doesn't clamp; that's the Servo's job.
        result = pico_controller._parse_command("P-10T200M1")
        assert result == (-10.0, 200.0, 1)


class TestServoMath:
    @pytest.mark.parametrize("angle, expected_us", [
        (0, 1000),
        (45, 1250),
        (90, 1500),
        (135, 1750),
        (180, 2000),
    ])
    def test_angle_to_pulse_us(self, pico_controller, angle, expected_us):
        duty = pico_controller._angle_to_duty_u16(angle)
        period_us = 1_000_000 // pico_controller.SERVO_FREQ_HZ
        pulse_us = duty * period_us / 65535
        assert abs(pulse_us - expected_us) < 5

    def test_below_zero_clamps(self, pico_controller):
        # _angle_to_duty_u16 clamps internally
        assert pico_controller._angle_to_duty_u16(-100) == pico_controller._angle_to_duty_u16(0)

    def test_above_180_clamps(self, pico_controller):
        assert pico_controller._angle_to_duty_u16(500) == pico_controller._angle_to_duty_u16(180)

    def test_duty_within_u16_range(self, pico_controller):
        for angle in range(0, 181, 5):
            duty = pico_controller._angle_to_duty_u16(angle)
            assert 0 <= duty <= 65535


class TestPicoHardwareWrappers:
    def test_servo_construct_and_write(self, pico_controller):
        servo = pico_controller.Servo(0)
        assert servo.angle == 90.0
        servo.write(45.0)
        assert servo.angle == 45.0
        servo.write(180.0)
        assert servo.angle == 180.0

    def test_servo_pwm_was_configured(self, pico_controller):
        servo = pico_controller.Servo(0)
        assert servo._pwm._freq == pico_controller.SERVO_FREQ_HZ
        assert len(servo._pwm.duty_history) >= 1
        servo.write(45.0)
        servo.write(170.0)
        assert servo._pwm.duty_history[-1] == pico_controller._angle_to_duty_u16(170.0)
