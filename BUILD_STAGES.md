# Build Stages — Electronics + Software, in Order

Each stage ends with a **GATE**: a test that must pass before you wire the next
subsystem. Never wire two new things at once — when something breaks you must
know which addition caused it.

Architecture and code reference: [AUTONOMY.md](AUTONOMY.md).

---

## Master GPIO map (fixed — everything below assumes this)

| Function | BCM | Physical pin | Notes |
|---|---|---|---|
| PCA9685 SDA | GPIO2 | 3 | I2C bus 1, addr `0x40` |
| PCA9685 SCL | GPIO3 | 5 | |
| iBUS in (FlySky RX) | GPIO15 (RXD) | 10 | `/dev/serial0`, receive-only |
| Sonar LEFT TRIG | GPIO5 | 29 | |
| Sonar LEFT ECHO | GPIO6 | 31 | **via divider** |
| Sonar CENTER TRIG | GPIO19 | 35 | |
| Sonar CENTER ECHO | GPIO26 | 37 | **via divider** |
| Sonar RIGHT TRIG | GPIO20 | 38 | |
| Sonar RIGHT ECHO | GPIO21 | 40 | **via divider** |
| GPS | — | USB port | USB-TTL adapter, NOT UART (iBUS owns it) |
| Webcam | — | USB 3.0 (blue) | |

PCA9685 channels: **ch0 = ESC (throttle)**, **ch1 = steering servo**.

## Power architecture (get this wrong and you fry things)

```
 LiPo ──► ESC ──► BLDC motor
           └─BEC 5-6V──► PCA9685 V+ (servo power ONLY)

 Power bank 5V/3A ──► Pi 4B USB-C          ← Pi NEVER runs off the ESC BEC

 ALL GROUNDS TIED TOGETHER: LiPo/ESC ── PCA9685 ── Pi ── sensors
```

Two rules that cause most RC-car Pi failures:
1. **Never power the Pi from the ESC's BEC.** Motor current spikes brown out the
   Pi; you'll chase "random reboots" that look like software bugs for a week.
2. **Common ground is mandatory.** PWM signals are meaningless without a shared
   reference. Missing ground = servo twitching or no response.

Verify power any time you add a load: `vcgencmd get_throttled` must print `0x0`.
Anything else means undervoltage (bad cable/bank) or thermal throttling.

---

# Stage 0 — Bench prep, no motion

**Electronics**
- [ ] Pi 4B: heatsink minimum, small fan recommended (₹200) — sustained inference
      thermally throttles a bare Pi 4.
- [ ] Flash **Raspberry Pi OS Lite 64-bit** (headless — the desktop costs ~400 MB
      of your 2 GB and you have none to spare). Enable SSH in Imager.
- [ ] Power bank → Pi. Confirm it delivers a genuine 5V/3A.

**Software**
```bash
sudo apt update && sudo apt install -y pigpio python3-venv i2c-tools
sudo systemctl enable --now pigpiod
sudo raspi-config          # enable I2C, enable serial PORT, disable serial CONSOLE
# zram instead of SD swap (SD swap destroys the card):
sudo apt install -y zram-tools
```

**GATE 0**: `vcgencmd get_throttled` → `0x0`, and `free -m` shows ≥ 1.6 GB free.

---

# Stage 1 — Drivetrain + RC override (you mostly have this)

The manual path must be bulletproof *before* any autonomy exists, because the RC
transmitter is your emergency stop for every later stage.

**Electronics**

| From | To |
|---|---|
| Pi GPIO2/GPIO3 | PCA9685 SDA/SCL |
| Pi 3.3V (pin 1) | PCA9685 VCC (logic) |
| ESC BEC 5-6V | PCA9685 **V+** (servo rail) |
| PCA9685 ch0 signal | ESC signal wire |
| PCA9685 ch1 signal | Steering servo signal |
| FlySky RX iBUS pin | Pi GPIO15 (pin 10) |
| everything | common GND |

Boot config for the iBUS UART — `/boot/firmware/config.txt`:
```
enable_uart=1
dtoverlay=disable-bt
```
then `sudo systemctl disable hciuart` and remove `console=serial0,115200` from
`cmdline.txt`. (The serial console will eat your iBUS bytes otherwise.)

**Software**
```bash
i2cdetect -y 1            # must show 40
donkey calibrate --channel 1 --bus 1   # steering: find left/right pulse values
donkey calibrate --channel 0 --bus 1   # throttle: find stop/fwd/rev pulse values
```
Put the measured numbers into `PWM_STEERING_THROTTLE` in `myconfig.py` — the
values currently in `config.py` are **placeholders and will not work**.

**GATE 1** — wheels **off the ground**, then on the ground:
- [ ] `python manage.py drive` — RC sticks move servo and motor correctly
- [ ] CH5 switch changes mode (`user` / `local_angle` / `local`) in the log
- [ ] Drive it around manually for 5 minutes. No reboots, no twitching.

---

# Stage 2 — Camera + the offline vision go/no-go (no new wiring on the car)

This is the cheapest stage to fail, so it comes before everything else.

**Electronics**
- [ ] USB webcam → Pi **USB 3.0 (blue)** port.
- [ ] Set `CAMERA_TYPE = "CVCAM"` in `myconfig.py` (the `WEBCAM` option uses
      pygame and is flaky on Linux).

**Software — on your laptop, in a venv:**
```bash
python -m venv .venv && .venv/Scripts/activate
pip install transformers torch onnx onnxruntime ultralytics
python scripts/export_models.py
```
Copy `exported_models/` → `mycar/models/` on the Pi.

**Collect footage**: walk every planned route holding the webcam at the car's
actual camera height and angle. Must include: sandy/unpaved stretches, at least
one speed breaker, people walking, and a junction/fork.

**Run the bench on the Pi:**
```bash
python scripts/vision_bench.py --video campus_walk.mp4 \
  --seg-model models/exported_models/segformer_sidewalk_int8.onnx \
  --seg-labels models/exported_models/segformer_labels.json \
  --yolo-model models/exported_models/yolov8n_ncnn_model --out annotated.mp4
```

**GATE 2** — watch `annotated.mp4` and read the printed numbers:
- Green mask covers the path you'd actually drive on, including sand? If not →
  fine-tune with ~100 labeled campus images on free Colab, re-export, re-run.
- Segmentation ≥ 1 FPS and peak RSS under ~1200 MB? If not → Pi 5 8 GB. Code
  needs no changes.
- Steering value swings the right direction as the path curves?

**Do not proceed until this gate passes.** Everything downstream assumes the mask
is trustworthy.

---

# Stage 3 — Segmentation pilot on the car, wheels off the ground

**Electronics**: none new. Mount the camera rigidly at a fixed height and tilt —
if it moves later, your corridor calibration is void.

**Calibration (do it once, properly)**
1. Tape a 1 m × 1 m grid on the ground in front of the car.
2. Capture a frame; note how many pixels wide your (car width + 20 cm margin)
   corridor spans at the **bottom** of the image.
3. `SEG_CORRIDOR_FRAC = that_width_px / image_width_px` in `myconfig.py`.

**Software**
```python
USE_CAMPUS_AUTONOMY = True     # everything else still False
CAMERA_TYPE = "CVCAM"
```

**GATE 3** — car up on a box, wheels spinning free:
- [ ] Point the camera at a clear path → wheels turn straight, motor spins slowly
- [ ] Block the view with cardboard → **throttle goes to zero**
- [ ] Angle the camera so the path is off to one side → steering follows it
- [ ] Flip CH5 to `user` mid-run → RC takes over instantly

---

# Stage 4 — HC-SR04 safety layer ⚠ voltage dividers required

**HC-SR04 ECHO outputs 5 V. Pi GPIO tolerates 3.3 V. Direct connection kills
the pin, possibly the Pi.** One divider per sensor, three total:

```
 ECHO ──[ 1 kΩ ]──┬── Pi GPIO (6 / 26 / 21)
                  │
                [ 2 kΩ ]
                  │
                 GND
```
5 V × 2k/(1k+2k) = **3.33 V**. Use 1 kΩ + 1.8 kΩ (→ 3.21 V) if you want margin.
TRIG needs no divider (Pi drives it at 3.3 V, which HC-SR04 accepts).

*Cleaner alternative:* buy **HC-SR04P** (~₹80) and power it at **3.3 V** — its
ECHO then swings to 3.3 V and no dividers are needed at all. Range drops to
~2–3 m, which is far more than the 80 cm we use.

**Mounting — this is the speed-breaker fix, don't skip it**
- [ ] Mount all three at **≥ 10–12 cm above the ground, aimed level or slightly
      up.** A 5–6 cm speed breaker then passes below the beam. Mounted low, the
      cart will hard-stop at every breaker forever and look broken.
- [ ] Splay left/right sensors ~30° outward so they don't hear each other's pings.
- [ ] VCC of all three → **5 V rail** (Pi pin 2/4 is fine, ~15 mA each).

**Software**
```python
HAVE_ULTRASONIC = True
ULTRASONIC_PINS = {"left": (5, 6), "center": (19, 26), "right": (20, 21)}
```
Standalone test first, before touching `manage.py`:
```bash
cd mycar && python -m parts.ultrasonic
```

**GATE 4**
- [ ] Standalone test prints sane cm values; wave your hand → numbers drop
- [ ] Full drive, wheels off ground: hand at < 30 cm → **throttle cuts**
- [ ] Remove hand → resumes after a moment (the 5-reading clear streak)
- [ ] Walk a speed breaker under the sensors → **no false stop**

---

# Stage 5 — First real autonomous drive

**Electronics**: none new.

**Software**: enable `HAVE_YOLO_GUARD = True`. Start conservative:
```python
SEG_THROTTLE_CRUISE = 0.20   # slower than default while tuning
SEG_THROTTLE_CREEP  = 0.14
```

**Procedure** — pick a straight, empty, obstacle-free path:
1. Walk **beside the car with the RC transmitter in your hand**, thumb on CH5.
2. Flip to `local`. Let it drive 10 m. Flip back to `user`.
3. Tune: car weaves → lower `SEG_KP`; car cuts corners → raise it. Oscillation
   that grows → raise `SEG_KD`.
4. Only after straights are stable, try a gentle curve.

**GATE 5**: 50 m of continuous autonomous driving with no intervention, then
deliberately step in front of it and confirm it stops.

---

# Stage 6 — Speed breakers and rough surfaces

**Physical gate — measure before you drive at one:**
- [ ] Wheel diameter ≥ ~2× breaker height (110–130 mm wheel for a 55–65 mm breaker)
- [ ] Belly clearance at **full suspension compression** clears the crest + 20 mm
- If either fails, that route gets marked impassable in the graph. Do not try to
  power over it — you'll beach the chassis or strip the 20T pinion.

**Software**: `HAVE_BREAKER_DETECT = True` (default on). Verify on your recorded
footage first — `vision_bench.py` prints `[BREAKER]` on the annotated frames.

**Optional MPU6050** (₹150) for *unpainted* humps, which have no stripes to
detect: wire to the **same I2C bus** as the PCA9685 (addr `0x68`, no conflict
with `0x40`) and use the pitch jolt to trigger creep.

**GATE 6**: cart approaches a striped breaker → drops to creep, crosses
perpendicular, resumes. Does not stop dead on it.

---

# Stage 7 — GPS, geofence, junctions

**Electronics**
- [ ] NEO-M8N + **active antenna** with clear sky view, mounted away from the ESC
      and Pi (both are RF-noisy). A small ground plane under the antenna helps.
- [ ] NEO-M8N → **USB-TTL adapter → Pi USB port.** Not the UART pins — iBUS owns
      `/dev/serial0` and moving it risks breaking your working RC link.
- [ ] GPS TX → adapter RX, GPS RX → adapter TX, VCC/GND per module spec.

**Software**
```bash
sudo apt install -y gpsd gpsd-clients
sudo dpkg-reconfigure gpsd     # device: /dev/ttyUSB0
cgps -s                        # confirm a 3D fix before anything else
```
On the laptop: `python scripts/build_campus_graph.py --point <lat>,<lon> --dist 800`,
copy `campus_graph.graphml` to `mycar/`.

```python
HAVE_GPS_NAV = True
GEOFENCE = [(lat1, lon1), (lat2, lon2), ...]   # WIDE margin — GPS error is 2-5 m
MISSION_REQUIRES_GPS = True                     # outdoor only; keep False indoors
```

**Survey first**: walk every route with `cgps` open. Tree cover and building
canyons give 10 m+ error or no fix at all. Routes with dead zones need either a
different path or buffered/replayed positions.

**GATE 7**
- [ ] Carry the powered car (motor disconnected) outside the geofence → log shows
      `GPS UNSAFE — fail closed`
- [ ] Drive a route with one junction; log shows `LEFT`/`RIGHT` at the right place
- [ ] Cart takes the correct branch at the fork

---

# Stage 8 — App, teleop, delivery

Per the existing design: FastAPI + SQLite + WebSocket backend, React + Leaflet
PWA, **Tailscale** to reach the Pi behind campus NAT, MediaMTX/WebRTC for the
operator video feed. Add the deadman heartbeat (command every 200 ms, cut
throttle after 500 ms of silence) before any remote driving.

Second camera (USB 720p for the operator, PiCam for the model) goes here — but
check RAM headroom first; on 2 GB this is the most likely OOM trigger.

---

## Shopping list (₹, approximate)

| Item | Cost | Stage |
|---|---|---|
| Resistors 1 kΩ + 2 kΩ (or HC-SR04P ×3) | 50 / 250 | 4 |
| Perfboard + jumper wires | 150 | 4 |
| Heatsink + fan for Pi | 200 | 0 |
| 5 V/3 A power bank (if current one is 2.1–2.4 A) | 1,000–1,500 | 0 |
| USB webcam | 800–1,500 | 2 |
| u-blox NEO-M8N + active antenna | 1,200–1,800 | 7 |
| USB-TTL adapter (CP2102/CH340) | 150 | 7 |
| MPU6050 (optional) | 150 | 6 |
| Multimeter (if you don't have one) | 500 | all |

**Upgrade trigger, not a purchase yet:** Pi 5 8 GB (~₹8,000) *only* if GATE 2
fails on RAM or speed. All wiring and code carry over unchanged.
