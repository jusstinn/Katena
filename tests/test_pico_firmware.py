"""Tests for the Pico W firmware (running under the `machine` shim).

We exercise the pure-logic functions (parser, servo math) directly.
The hardware interaction layers (Servo, Ultrasonic, StatusOutput) are
exercised through the shim to verify they at least construct and don't
explode.
"""

from __future__ import annotations

import pytest


class TestPicoCommandParser:
    def test_valid_command_no_rotation(self, pico_controller):
        assert pico_controller._parse_command("P90.0T45.5M2") == (90.0, 45.5, None, 2)

    def test_integer_angles(self, pico_controller):
        assert pico_controller._parse_command("P90T45M0") == (90.0, 45.0, None, 0)

    def test_zero_and_extreme_angles(self, pico_controller):
        assert pico_controller._parse_command("P0.0T180.0M3") == (0.0, 180.0, None, 3)

    def test_with_rotation(self, pico_controller):
        assert pico_controller._parse_command("P90T90R45.0M1") == (90.0, 90.0, 45.0, 1)

    def test_with_negative_rotation(self, pico_controller):
        assert pico_controller._parse_command("P90T90R-90M1") == (90.0, 90.0, -90.0, 1)

    def test_rotation_zero_explicit(self, pico_controller):
        assert pico_controller._parse_command("P90T90R0.0M0") == (90.0, 90.0, 0.0, 0)

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
        "P90T90RxxM0",
    ])
    def test_invalid_inputs_return_none(self, pico_controller, bad):
        assert pico_controller._parse_command(bad) is None

    def test_trailing_whitespace_tolerated(self, pico_controller):
        assert pico_controller._parse_command("P90T90M0  ") == (90.0, 90.0, None, 0)
        assert pico_controller._parse_command("P90T90R10M0  ") == (90.0, 90.0, 10.0, 0)

    def test_negative_angles_pass_through(self, pico_controller):
        # Parser doesn't clamp; that's the Servo's / Stepper's job.
        result = pico_controller._parse_command("P-10T200M1")
        assert result == (-10.0, 200.0, None, 1)


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


class TestStepperMath:
    @pytest.mark.parametrize("deg, expected_steps", [
        (0.0, 0),
        (360.0, 4096),
        (90.0, 1024),
        (-90.0, -1024),
        (180.0, 2048),
    ])
    def test_deg_to_steps(self, pico_controller, deg, expected_steps):
        assert pico_controller._deg_to_steps(deg) == expected_steps

    def test_round_trip(self, pico_controller):
        for deg in (-180.0, -45.0, 0.0, 1.0, 45.0, 90.0, 180.0):
            steps = pico_controller._deg_to_steps(deg)
            back = pico_controller._steps_to_deg(steps)
            assert abs(back - deg) <= pico_controller.STEPPER_DEG_PER_STEP

    def test_half_step_sequence_is_8_unique_patterns(self, pico_controller):
        seq = pico_controller.STEPPER_HALF_STEP_SEQ
        assert len(seq) == 8
        assert len(set(seq)) == 8


class TestStepper28BYJ48:
    def test_idle_starts_de_energized(self, pico_controller):
        st = pico_controller.Stepper28BYJ48(6, 7, 8, 9)
        for pin in st._pins:
            assert pin.value() == 0
        assert st.angle_deg == 0.0
        assert not st.is_moving

    def test_set_target_then_tick_advances_position(self, pico_controller):
        st = pico_controller.Stepper28BYJ48(6, 7, 8, 9)
        st.set_target_deg(10.0)
        assert st.is_moving
        # Force the rate-limit not to apply by zeroing last_step_us
        for _ in range(200):
            st._last_step_us = 0
            st.tick()
        # Should have stepped repeatedly toward target (positive direction)
        assert st._current_steps > 0

    def test_target_reached_eventually_releases_coils(self, pico_controller):
        st = pico_controller.Stepper28BYJ48(6, 7, 8, 9)
        st.set_target_deg(0.5)
        for _ in range(20):
            st._last_step_us = 0
            st.tick()
        assert not st.is_moving
        # Simulate IDLE_RELEASE_MS having elapsed
        import time as _time
        st._reached_target_at_ms = _time.ticks_ms() - pico_controller.STEPPER_IDLE_RELEASE_MS - 1
        st.tick()
        for pin in st._pins:
            assert pin.value() == 0

    def test_negative_target_steps_backward(self, pico_controller):
        st = pico_controller.Stepper28BYJ48(6, 7, 8, 9)
        st.set_target_deg(-10.0)
        for _ in range(200):
            st._last_step_us = 0
            st.tick()
        assert st._current_steps < 0
