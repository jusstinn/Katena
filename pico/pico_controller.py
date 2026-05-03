"""Katena pan/tilt + sensor controller for Raspberry Pi Pico W.

Runs on MicroPython. Upload as `main.py` to auto-start on boot.

Pin map (see pico/README.md for wiring + power notes):
    GP0   pan servo PWM      (50 Hz)
    GP1   tilt servo PWM     (50 Hz)
    GP2   HC-SR04 trigger    (output)
    GP3   HC-SR04 echo       (input, voltage-divided to 3.3V)
    GP14  buzzer             (active, drive HIGH)
    GP15  status LED
    GP26  LDR / photodetector (ADC0)

Serial protocol (115200 baud over USB):
    Host -> Pico:   "P{pan}T{tilt}M{mode}\\n"
                    pan/tilt 0-180 degrees, mode 0-3
    Pico -> Host:   "D{cm}S{status}L{ldr}\\n"
                    distance cm (-1 if invalid), status 0-3, LDR 0-1023

Modes:
    0 = idle              LED off,    no buzzer
    1 = tracking          LED green,  no buzzer
    2 = sweep             LED amber,  no buzzer
    3 = locked / firing   LED red,    buzzer pulsed when fiber compromised
"""

import select
import sys
import time

from machine import ADC, PWM, Pin

PAN_PIN = 0
TILT_PIN = 1
TRIG_PIN = 2
ECHO_PIN = 3
BUZZER_PIN = 14
LED_PIN = 15
LDR_ADC_CHAN = 26

SERVO_FREQ_HZ = 50
SERVO_MIN_US = 1000
SERVO_MAX_US = 2000
SERVO_PERIOD_US = 1_000_000 // SERVO_FREQ_HZ

TELEMETRY_INTERVAL_MS = 100
ULTRASONIC_INTERVAL_MS = 200
ULTRASONIC_TIMEOUT_US = 30000

LDR_FIBER_FULL = 900
LDR_FIBER_CUT = 50


def _angle_to_duty_u16(angle):
    if angle < 0.0:
        angle = 0.0
    elif angle > 180.0:
        angle = 180.0
    pulse_us = SERVO_MIN_US + (angle / 180.0) * (SERVO_MAX_US - SERVO_MIN_US)
    return int(pulse_us * 65535 / SERVO_PERIOD_US)


class Servo:
    def __init__(self, pin_no):
        self._pwm = PWM(Pin(pin_no))
        self._pwm.freq(SERVO_FREQ_HZ)
        self._angle = 90.0
        self.write(90.0)

    def write(self, angle):
        self._angle = angle
        self._pwm.duty_u16(_angle_to_duty_u16(angle))

    @property
    def angle(self):
        return self._angle


class Ultrasonic:
    def __init__(self, trig_pin, echo_pin):
        self._trig = Pin(trig_pin, Pin.OUT)
        self._echo = Pin(echo_pin, Pin.IN)
        self._trig.value(0)
        self._last_cm = -1.0
        self._last_read_ms = 0

    def read_cm(self, force=False):
        now = time.ticks_ms()
        if not force and time.ticks_diff(now, self._last_read_ms) < ULTRASONIC_INTERVAL_MS:
            return self._last_cm
        self._last_read_ms = now

        self._trig.value(0)
        time.sleep_us(2)
        self._trig.value(1)
        time.sleep_us(10)
        self._trig.value(0)

        t0 = time.ticks_us()
        while self._echo.value() == 0:
            if time.ticks_diff(time.ticks_us(), t0) > ULTRASONIC_TIMEOUT_US:
                self._last_cm = -1.0
                return -1.0
        start = time.ticks_us()
        while self._echo.value() == 1:
            if time.ticks_diff(time.ticks_us(), start) > ULTRASONIC_TIMEOUT_US:
                self._last_cm = -1.0
                return -1.0
        end = time.ticks_us()

        echo_us = time.ticks_diff(end, start)
        cm = (echo_us / 2.0) * 0.0343
        if cm > 400.0 or cm < 2.0:
            self._last_cm = -1.0
        else:
            self._last_cm = cm
        return self._last_cm


class StatusOutput:
    def __init__(self, led_pin, buzzer_pin):
        self._led = Pin(led_pin, Pin.OUT)
        self._buzzer = Pin(buzzer_pin, Pin.OUT)
        self._buzzer.value(0)
        self._buzz_until_ms = 0

    def update(self, mode, fiber_compromised):
        if mode == 0:
            self._led.value(0)
        elif mode == 3:
            self._led.value((time.ticks_ms() // 100) % 2)
        else:
            self._led.value(1)

        if fiber_compromised:
            self._buzz_until_ms = time.ticks_add(time.ticks_ms(), 500)
        if time.ticks_diff(self._buzz_until_ms, time.ticks_ms()) > 0:
            self._buzzer.value((time.ticks_ms() // 80) % 2)
        else:
            self._buzzer.value(0)


def _parse_command(line):
    """Parse 'P{pan}T{tilt}M{mode}'. Returns (pan, tilt, mode) or None."""
    try:
        if "P" not in line or "T" not in line or "M" not in line:
            return None
        p_idx = line.index("P")
        t_idx = line.index("T", p_idx)
        m_idx = line.index("M", t_idx)
        pan = float(line[p_idx + 1 : t_idx])
        tilt = float(line[t_idx + 1 : m_idx])
        mode = int(line[m_idx + 1 :].strip())
        if mode < 0 or mode > 3:
            return None
        return pan, tilt, mode
    except (ValueError, IndexError):
        return None


def main():
    pan_servo = Servo(PAN_PIN)
    tilt_servo = Servo(TILT_PIN)
    ultrasonic = Ultrasonic(TRIG_PIN, ECHO_PIN)
    status_io = StatusOutput(LED_PIN, BUZZER_PIN)
    ldr = ADC(LDR_ADC_CHAN)

    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)

    mode = 0
    last_telemetry_ms = 0
    rx_buffer = ""
    last_signal = 1.0

    while True:
        events = poll.poll(0)
        if events:
            ch = sys.stdin.read(1)
            if ch:
                rx_buffer += ch
                if "\n" in rx_buffer:
                    line, _, rx_buffer = rx_buffer.partition("\n")
                    parsed = _parse_command(line.strip())
                    if parsed is not None:
                        pan, tilt, mode = parsed
                        pan_servo.write(pan)
                        tilt_servo.write(tilt)

        now = time.ticks_ms()
        if time.ticks_diff(now, last_telemetry_ms) >= TELEMETRY_INTERVAL_MS:
            last_telemetry_ms = now
            cm = ultrasonic.read_cm()
            ldr_raw_16 = ldr.read_u16()
            ldr_raw = ldr_raw_16 >> 6
            span = max(1, LDR_FIBER_FULL - LDR_FIBER_CUT)
            signal = max(0.0, min(1.0, (ldr_raw - LDR_FIBER_CUT) / span))
            fiber_compromised = (
                signal < 0.5 and last_signal >= 0.5 and mode == 3
            )
            last_signal = signal

            status_io.update(mode, fiber_compromised)

            cm_out = -1 if cm is None or cm < 0 else cm
            print("D{:.1f}S{:d}L{:d}".format(cm_out, mode, ldr_raw))


if __name__ == "__main__":
    main()
