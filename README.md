# Katena — FOG Drone Fiber-Tether Neutralization System

> **One sentence:** Jammers don't work on Fiber-Optic-Guided drones.
> We don't jam — we corrupt the optical channel by damaging the fiber's
> cladding with a precision laser, dropping the control link in
> milliseconds.
>
> National Security Hackathon 2026 · Palantir track · Built on CASK.

---

## The thesis

**RF silence is a positive identifier of a FOG drone.** They're immune
to jamming, GPS spoofing, and high-power microwave because they carry no
RF signature — that's the whole point of fiber. The defense industry's
current frontline answer to FOG drones in Ukraine is *literal barbed
wire stretched across fields.* We do better.

The novel insight: **you don't have to sever the fiber.** Total internal
reflection in the core depends on an intact cladding. Breach the
protective jacket, scratch the cladding, and the optical channel
immediately bleeds light out and ambient light in. Bit-error-rate spikes
past the FEC threshold, control link drops, drone enters failsafe.
A laser pointer is enough.

## How the system works

```
                ┌───────────────────────────────────┐
                │  Camera (USB) on pan/tilt gimbal  │
                └─────────────┬─────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  Detection: OpenCV motion + YOLOv8      │
        │  ensemble. Output: drone bounding box.  │
        └─────────────┬───────────────────────────┘
                      ▼
        ┌─────────────────────────────────────────┐
        │  Aim point = bbox bottom-center,        │
        │  offset down 10–20% (fiber exit zone    │
        │  under the motors).                     │
        └─────────────┬───────────────────────────┘
                      ▼
        ┌─────────────────────────────────────────┐
        │  Pixel → servo angle (calibration map). │
        │  Send "P{pan}T{tilt}M{mode}" to Pico.   │
        └─────────────┬───────────────────────────┘
                      ▼
        ┌─────────────────────────────────────────┐
        │  Pico W drives PWM. Photodetector       │
        │  reports fiber signal level back.       │
        └─────────────┬───────────────────────────┘
                      ▼
        ┌─────────────────────────────────────────┐
        │  Engagement logged as DroneEngagement   │
        │  ontology object → Foundry (when up) +  │
        │  local JSONL (always).                  │
        └─────────────────────────────────────────┘
```

The same stack runs on a Palantir CASK (Jetson) at the edge — fully
autonomous when the network is down, syncs to Foundry when it's up.

## Repository layout

```
Katena/
├── .venv/                    # Local Python 3.11 venv (uv-managed)
├── .venv-jetson/             # Created on the Jetson by setup_jetson.sh
├── .env                      # Foundry creds (gitignored — copy from .env.example)
├── requirements.txt          # Frozen dep set (72 packages)
├── yolov8n.pt                # Pretrained YOLOv8 weights
│
├── macbook/                  # Tracking + calibration + tracker UI (TBD)
├── pico/                     # MicroPython servo controller
│   ├── firmware/             # MicroPython .uf2 ready to flash
│   └── README.md             # Flashing + serial protocol
├── jetson/                   # CASK / Jetson bootstrap and verification
│   ├── setup_jetson.sh       # One-shot Jetson environment setup
│   ├── verify_jetson.py      # Hardware/driver smoke test
│   └── README.md             # SSH + transfer + troubleshooting
├── scripts/                  # Cross-cutting verification + helpers
│   ├── verify_camera.py
│   ├── verify_yolo.py
│   ├── verify_serial.py
│   ├── verify_foundry.py
│   ├── verify_all.py         # Run all verifications before each demo
│   ├── seed_engagements.py   # Pre-mint demo data for dashboard
│   └── run_tests.sh          # Sanity gate: compile + tests + (optional) smoke
├── tests/                    # 123 unit tests, ~3s, no hardware needed
│   ├── conftest.py           # MicroPython `machine` shim + shared fixtures
│   ├── test_engagement.py
│   ├── test_state_machine.py
│   ├── test_logger.py
│   ├── test_calibration.py
│   ├── test_config.py
│   ├── test_detector.py
│   ├── test_overlay.py
│   ├── test_serial_link.py
│   └── test_pico_firmware.py
├── dashboard.py              # Streamlit local twin of Foundry view
├── Makefile                  # `make test`, `make tracker`, etc.
├── pyproject.toml            # pytest + coverage config
├── downloads/                # Local installers (Thonny .pkg) — gitignored
└── paper/                    # LaTeX physics writeup (cladding-breach math)
```

## Quickstart (MacBook)

Already done — environment is live. To activate from a fresh shell:

```bash
cd ~/Desktop/Katena
source .venv/bin/activate
```

Daily commands:

```bash
make test           # 123 unit tests, ~3s, no hardware needed
make tracker        # run the live tracker in mock Pico mode
make calibrate      # run the calibration teleop tool
make dashboard      # open the Streamlit ops dashboard
make seed           # pre-seed dashboard with demo engagements
make smoke          # camera + YOLO + serial + Foundry live checks
make help           # all available commands
```

Before each commit, run:

```bash
bash scripts/run_tests.sh
```

This runs syntax check + 123 unit tests in under 5 seconds. Add
`--smoke` to also exercise camera/YOLO/serial/Foundry live.

### Configure Foundry

```bash
cp .env.example .env
# Edit .env: fill in FOUNDRY_URL and FOUNDRY_TOKEN from Developer Console
python scripts/verify_foundry.py
```

### Flash the Pico W

See `pico/README.md`. tl;dr: hold BOOTSEL, plug in USB, drag
`pico/firmware/RPI_PICO_W-20260406-v1.28.0.uf2` onto the `RPI-RP2`
drive that appears.

### Set up the CASK / Jetson

See `jetson/README.md`. tl;dr: SSH in, `rsync` the project over,
`bash jetson/setup_jetson.sh`, then `python3 jetson/verify_jetson.py`.

## Tech stack

| Layer | Tool |
| --- | --- |
| Compute (primary) | MacBook Air M5 (Apple Silicon, Metal/MPS) |
| Compute (edge) | Palantir CASK on NVIDIA Jetson |
| Vision | OpenCV 4.13 + YOLOv8n (Ultralytics 8.4) |
| Servo control | Pico W running MicroPython 1.28 |
| Ontology / sync | Palantir Foundry + OSDK (Embedded Ontology) |
| Local dashboard | Streamlit |
| Optional sensor | RTL-SDR for RF-silence FOG confirmation |

## Demo flow

1. Suspended drone with POF tether visible to photodetector.
2. Live camera + state-machine UI: `SEARCHING` → `TARGET ACQUIRED` →
   `RF SILENCE — FOG CONFIRMED` → `ENGAGING`.
3. Servos auto-aim, laser dot lands at fiber-exit zone under the motors.
4. Pre-recorded close-up of the laser hitting the fiber + signal
   monitor degrading 100 % → 0 %.
5. Live: `FIBER COMPROMISED` flashes, buzzer triggers,
   `DroneEngagement` object appears on Foundry Workshop dashboard.
6. **Network kill switch.** System keeps detecting and engaging.
7. Restore network. Queued offline engagements sync to Foundry.

## Status

| Component | State |
| --- | --- |
| Python env, deps, YOLO weights | ✅ Done |
| Project scaffolding + verification scripts | ✅ Done |
| MicroPython firmware downloaded | ✅ Done |
| Thonny installer downloaded | ✅ Done (run installer from `downloads/`) |
| Camera permission granted in macOS | ⏳ User action required |
| Foundry: object types defined | ⏳ Tomorrow |
| Foundry: OSDK client generated | ⏳ Tomorrow |
| Pico W flashed | ⏳ When hardware arrives |
| `macbook/main_tracker.py` (detection + servo cmd) | ⏳ Build phase |
| `pico/pico_controller.py` (servo + sensor firmware) | ⏳ Build phase |
| Jetson bootstrap | ⏳ When CASK kit is in hand |
| `paper/main.tex` (physics writeup) | ⏳ Optional |
