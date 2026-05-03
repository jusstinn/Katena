# Pico W — flash and dev workflow

The Pico W is the dumb-but-fast PWM controller for the pan/tilt servos.
The MacBook (or Jetson) does all the brains; the Pico just translates
serial commands into PWM and reports sensor readings back.

## 1. Flash MicroPython

The firmware is already downloaded at:

```
pico/firmware/RPI_PICO_W-20260406-v1.28.0.uf2
```

To flash:

1. Hold the **BOOTSEL** button on the Pico W while plugging it into USB.
2. It mounts as a USB drive named `RPI-RP2`.
3. Drag `pico/firmware/RPI_PICO_W-20260406-v1.28.0.uf2` onto that drive.
4. The drive auto-ejects and the Pico reboots into MicroPython.

After reboot, it appears as a serial device (likely `/dev/cu.usbmodem101`
on macOS — verify with `python scripts/verify_serial.py`).

## 2. Two ways to upload code

### Option A — Thonny (GUI, friendlier)

The installer is at `downloads/thonny-5.0.0-arm64.pkg`. Double-click,
install, then in Thonny:

1. Tools → Options → Interpreter → choose **MicroPython (Raspberry Pi
   Pico)** and pick the right serial port.
2. Open `pico/pico_controller.py` (when it exists), File → Save As →
   choose **Raspberry Pi Pico** → save as `main.py`. Anything saved as
   `main.py` runs automatically on boot.

### Option B — mpremote (CLI, faster iteration)

Already installed in the venv. From the project root:

```bash
source .venv/bin/activate

# List connected devices
mpremote connect list

# Copy a file to the Pico
mpremote cp pico/pico_controller.py :main.py

# Run a file once without saving
mpremote run pico/pico_controller.py

# Open a REPL (Ctrl-X to exit)
mpremote repl

# Reset the Pico
mpremote reset
```

Tip: alias `pico='mpremote'` in your shell to type less.

## 3. Wiring (when hardware arrives)

Reference for the controller code we'll write next:

| Pin | Signal | Notes |
| --- | --- | --- |
| GP0 | Pan servo PWM | 50 Hz, 1–2 ms pulse |
| GP1 | Tilt servo PWM | 50 Hz, 1–2 ms pulse |
| GP2 | HC-SR04 trigger | 5V tolerant, 10 µs pulse |
| GP3 | HC-SR04 echo | Voltage divider down to 3.3V! |
| GP14 | Buzzer | Active buzzer, drive HIGH to sound |
| GP15 | Status LED (red/green common) | PWM for brightness if RGB |
| GP26 (ADC0) | LDR / photodetector | Fiber signal level |
| 5V (VBUS) | Servo + sensor power | **External 5V battery, NOT VSYS** |
| GND | Common ground | Tie Pico GND, servo GND, sensor GND together |

**Power note:** 9G servos can draw 600–700 mA peak. Do NOT power them
from the Pico's 3.3V rail or from the Pico's USB rail directly. Use a
separate 5V battery (4xAA, USB power bank, or 5V buck converter from a
LiPo) and tie its ground to the Pico's ground.

## 4. Serial protocol (planned)

```
Host -> Pico:    "P{pan}T{tilt}M{mode}\n"
                 P = pan angle 0-180
                 T = tilt angle 0-180
                 M = mode (0=idle, 1=tracking, 2=sweep, 3=locked)

Pico -> Host:    "D{distance_cm}S{status}L{ldr}\n"
                 D = ultrasonic distance, cm
                 S = status (0=idle, 1=tracking, 2=sweeping, 3=lost)
                 L = LDR / photodetector reading (0-1023)
```

Both sides at 115200 baud, line-terminated. Simple to debug with `screen`
or `mpremote repl`.

## 5. Quick troubleshooting

| Symptom | Fix |
| --- | --- |
| Pico doesn't show up as serial | Re-flash the .uf2 firmware. BOOTSEL while plugging in. |
| Servos jitter / brown out | External 5V power, common ground. NOT from VSYS. |
| `mpremote: no devices found` | Try `mpremote connect /dev/cu.usbmodem101` explicitly. |
| HC-SR04 returns garbage | The echo pin is 5V — needs a voltage divider to 3.3V before GP3, or the Pico will eventually fry. |
