"""Katena pan/tilt + base-rotation + sensor controller for Raspberry Pi Pico W.

Runs on MicroPython. Upload as `main.py` to auto-start on boot.

Pin map (see pico/README.md for wiring + power notes):
    GP0   tilt servo PWM      (50 Hz, hobby servo)  -- swapped from
    GP1   pan servo PWM       (50 Hz, hobby servo)     the original wiring
    GP2   HC-SR04 trigger     (output)
    GP3   HC-SR04 echo        (input, voltage-divided to 3.3V)
    GP4   PIR / Fresnel       (input, internal pull-down so an
                               unwired sensor reads safely as 0)
    GP6   stepper IN1         (28BYJ-48 via ULN2003)
    GP7   stepper IN2
    GP8   stepper IN3
    GP9   stepper IN4
    GP14  buzzer              (active, drive HIGH)
    GP15  status LED
    GP26  LDR / photodetector (ADC0)

Serial protocol (115200 baud over USB):
    Host -> Pico:   "P{pan}T{tilt}R{rot}M{mode}\\n"
                    pan/tilt 0-180 degrees, rot in degrees (relative to
                    power-on position), mode 0-3.
                    R is OPTIONAL — if absent the stepper target is
                    unchanged (older host code keeps working).
    Pico -> Host:   "D{cm}S{status}L{ldr}A{rot}P{pir}\\n"
                    distance cm (-1 if invalid), status 0-3,
                    LDR 0-1023, A = current stepper angle in degrees,
                    P = PIR digital state (0 = clear, 1 = motion).
                    P is APPENDED to keep older parsers (which only
                    look up to A) working unchanged.

Modes:
    0 = idle              LED off,    no buzzer, stepper coils de-energized
    1 = tracking          LED green,  no buzzer
    2 = sweep             LED amber,  no buzzer
    3 = locked / firing   LED red,    buzzer pulsed when fiber compromised

Stepper notes:
    The 28BYJ-48 is a unipolar 5V geared stepper with a 64:1 internal
    gearbox. Driven in half-step mode (8 sub-steps per electrical cycle)
    that's 4096 sub-steps per output-shaft revolution, i.e.
    ~11.378 sub-steps per degree, ~0.088 deg per sub-step.
    Drive it at ~660 sub-steps/sec (1500 us per step) for smooth motion
    well within the motor's safe speed envelope (~15 RPM max).
    When the target is reached and we're in IDLE for a moment, all four
    coils are switched off so the motor stops drawing ~250 mA holding
    current and stops getting hot.
"""

import select
import sys
import time

from machine import ADC, PWM, Pin

PAN_PIN = 1
TILT_PIN = 0
TRIG_PIN = 2
ECHO_PIN = 3
PIR_PIN = 4
STEPPER_IN1 = 6
STEPPER_IN2 = 7
STEPPER_IN3 = 8
STEPPER_IN4 = 9
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

STEPPER_STEPS_PER_REV = 4096
STEPPER_DEG_PER_STEP = 360.0 / STEPPER_STEPS_PER_REV
STEPPER_STEP_INTERVAL_US = 1500
STEPPER_IDLE_RELEASE_MS = 400
STEPPER_HALF_STEP_SEQ = (
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
)


def _angle_to_duty_u16(angle):
    if angle < 0.0:
        angle = 0.0
    elif angle > 180.0:
        angle = 180.0
    pulse_us = SERVO_MIN_US + (angle / 180.0) * (SERVO_MAX_US - SERVO_MIN_US)
    return int(pulse_us * 65535 / SERVO_PERIOD_US)


def _deg_to_steps(deg):
    return int(round(deg / STEPPER_DEG_PER_STEP))


def _steps_to_deg(steps):
    return steps * STEPPER_DEG_PER_STEP


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


class Stepper28BYJ48:
    """Non-blocking half-step driver for a 28BYJ-48 + ULN2003.

    Track absolute step position relative to power-on (which is treated
    as 0 deg). Move towards `target_steps` one micro-step at a time,
    cooperatively, when `tick()` is called from the main loop.

    De-energizes all four coils after `STEPPER_IDLE_RELEASE_MS` of
    sitting at the target so the motor doesn't cook itself drawing
    holding current.
    """

    def __init__(self, pin_in1, pin_in2, pin_in3, pin_in4):
        self._pins = (
            Pin(pin_in1, Pin.OUT),
            Pin(pin_in2, Pin.OUT),
            Pin(pin_in3, Pin.OUT),
            Pin(pin_in4, Pin.OUT),
        )
        self._current_steps = 0
        self._target_steps = 0
        self._seq_index = 0
        self._last_step_us = time.ticks_us()
        self._reached_target_at_ms = time.ticks_ms()
        self._energized = False
        self._release()

    def _release(self):
        for pin in self._pins:
            pin.value(0)
        self._energized = False

    def _apply_pattern(self):
        pattern = STEPPER_HALF_STEP_SEQ[self._seq_index]
        for pin, val in zip(self._pins, pattern):
            pin.value(val)
        self._energized = True

    def set_target_deg(self, deg):
        self._target_steps = _deg_to_steps(deg)
        if self._target_steps != self._current_steps:
            self._reached_target_at_ms = time.ticks_ms()

    @property
    def angle_deg(self):
        return _steps_to_deg(self._current_steps)

    @property
    def is_moving(self):
        return self._current_steps != self._target_steps

    def tick(self):
        now_us = time.ticks_us()
        if self._current_steps == self._target_steps:
            if (
                self._energized
                and time.ticks_diff(time.ticks_ms(), self._reached_target_at_ms)
                >= STEPPER_IDLE_RELEASE_MS
            ):
                self._release()
            return
        if time.ticks_diff(now_us, self._last_step_us) < STEPPER_STEP_INTERVAL_US:
            return
        self._last_step_us = now_us

        if self._target_steps > self._current_steps:
            self._seq_index = (self._seq_index + 1) % 8
            self._current_steps += 1
        else:
            self._seq_index = (self._seq_index - 1) % 8
            self._current_steps -= 1
        self._apply_pattern()
        self._reached_target_at_ms = time.ticks_ms()


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
    """Parse 'P{pan}T{tilt}[R{rot}]M{mode}'.

    R is optional. Returns (pan, tilt, rotation_or_None, mode) or None
    on a malformed line. Older 'P{pan}T{tilt}M{mode}' commands keep
    working — rotation comes back as None and the caller leaves the
    stepper target unchanged.
    """
    try:
        if "P" not in line or "T" not in line or "M" not in line:
            return None
        p_idx = line.index("P")
        t_idx = line.index("T", p_idx)
        m_idx = line.index("M", t_idx)
        r_idx = line.find("R", t_idx, m_idx)

        pan_end = r_idx if r_idx != -1 else m_idx
        pan = float(line[p_idx + 1 : t_idx])
        tilt = float(line[t_idx + 1 : pan_end])
        rotation = float(line[r_idx + 1 : m_idx]) if r_idx != -1 else None
        mode = int(line[m_idx + 1 :].strip())
        if mode < 0 or mode > 3:
            return None
        return pan, tilt, rotation, mode
    except (ValueError, IndexError):
        return None


def main():
    pan_servo = Servo(PAN_PIN)
    tilt_servo = Servo(TILT_PIN)
    stepper = Stepper28BYJ48(STEPPER_IN1, STEPPER_IN2, STEPPER_IN3, STEPPER_IN4)
    ultrasonic = Ultrasonic(TRIG_PIN, ECHO_PIN)
    status_io = StatusOutput(LED_PIN, BUZZER_PIN)
    ldr = ADC(LDR_ADC_CHAN)
    # PIR / Fresnel proximity sensor. Internal pull-down so a not-yet-
    # connected pin reads safely as 0 (no spurious "motion" before the
    # sensor is wired). Most HC-SR501-style modules drive the OUT line
    # actively, so the pull-down doesn't fight them once connected.
    pir = Pin(PIR_PIN, Pin.IN, Pin.PULL_DOWN)

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
                        pan, tilt, rotation, mode = parsed
                        pan_servo.write(pan)
                        tilt_servo.write(tilt)
                        if rotation is not None:
                            stepper.set_target_deg(rotation)

        stepper.tick()

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
            pir_state = pir.value()
            print(
                "D{:.1f}S{:d}L{:d}A{:.1f}P{:d}".format(
                    cm_out, mode, ldr_raw, stepper.angle_deg, pir_state
                )
            )


if __name__ == "__main__":
    main()
