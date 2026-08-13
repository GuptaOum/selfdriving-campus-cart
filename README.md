# Self-Driving RC Car — Campus Delivery Cart Prototype

An autonomous RC car built on the [DonkeyCar](https://www.donkeycar.com/) platform, using behavioral cloning (end-to-end deep learning) to drive itself. 

> **⚠️ This is a prototype.** The goal of this stage is simply to prove that self-driving works end-to-end on cheap hardware. The long-term vision is to scale this into an **autonomous point-A-to-point-B campus delivery cart system** — a small vehicle that can carry items (documents, parcels, food) between buildings on a campus without a driver.

## Vision & Roadmap

| Phase | Goal | Approach | Status |
|-------|------|----------|--------|
| **1. Track autonomy** | Car drives itself around indoor tracks with obstacles; train and compare models across 5–6 different track layouts | Behavioral cloning (DonkeyCar) | 🔄 **In progress — current focus** |
| **2. Campus delivery** | Point-A-to-point-B outdoor navigation on campus paths, obstacle avoidance, live tracking app | Pretrained segmentation + geometric steering | 🧊 Code written, dormant behind a config flag |
| 3. AI commands | Natural-language commands ("go to the parking area") via object detection + LLM command parsing | YOLO + LLM | 📋 Deferred, not cancelled |

Phase 1 is the priority and is unaffected by Phase 2 — see below.

## How It Works

DonkeyCar uses **behavioral cloning**: you drive the car manually around a track while it records camera frames paired with your steering/throttle inputs. A convolutional neural network (based on NVIDIA's end-to-end self-driving architecture) is then trained to imitate your driving. At inference time, the Raspberry Pi feeds live camera frames to the model, which outputs steering and throttle commands ~20 times per second.

```
Drive manually → Record data → Train CNN → Car drives itself
```

## Two driving modes (they do not interfere)

The repo contains two completely independent ways to drive the car. Only one is
active at a time, chosen by a single flag in `myconfig.py`.

| | **Phase 1 — Behavioral cloning** | **Phase 2 — Pretrained vision** |
|---|---|---|
| Flag | `USE_CAMPUS_AUTONOMY = False` *(default)* | `USE_CAMPUS_AUTONOMY = True` |
| Needs training data | Yes — you drive and record | No — models are pretrained |
| Steering comes from | A CNN trained on your driving | Geometry over a drivable-area mask |
| Best for | Repeatable tracks, model comparison | Open campus paths, unseen routes |
| Code | stock DonkeyCar + `ibus_receiver.py` | `mycar/parts/` |

**The default is Phase 1.** With `USE_CAMPUS_AUTONOMY = False`, `manage.py` behaves
exactly like stock DonkeyCar — record tubs, `train.py`, then
`manage.py drive --model models/mypilot.h5`. Nothing in `mycar/parts/` loads or runs.

### Phase 1 plan: multi-track training

Train and evaluate across 5–6 different track layouts with obstacles, to measure how
well a cloned model generalizes rather than memorizes:

- Record a tub per track, keeping layouts visually distinct.
- Train per-track models *and* one combined model; compare lap success rates.
- Ablations worth running: `linear` vs `categorical` model type, and
  `ROI_CROP_TOP` on vs off — cropping the top of the frame removes track-specific
  background and is the single biggest anti-overfitting lever.
- Set `CREATE_TF_LITE = True`. Training happens on x86, inference on ARM, and
  mismatched TensorFlow versions are the most common "model won't load" failure.

## Hardware Architecture

```
FlySky Transmitter ──RF──▶ FlySky Receiver ──iBUS/UART──▶ Raspberry Pi ──I2C──▶ PCA9685 ──PWM──▶ Servo (steering)
                                                              │                              └───▶ ESC (throttle)
                                                          Pi Camera
```

- **FlySky TX/RX** — manual control and manual/auto mode switching. The receiver talks to the Pi over the **iBUS protocol** (digital serial, all 14 channels over one wire — much cleaner than reading raw PWM).
- **Raspberry Pi** — runs the DonkeyCar vehicle loop, reads iBUS, runs the trained model in autopilot mode.
- **PCA9685** — 16-channel I2C PWM driver. Generates stable, jitter-free PWM for the servo and ESC (direct GPIO PWM is too jittery for smooth steering).
- **Servo + ESC** — powered from an external battery rail; **all components share a common ground**.
- **Camera** — front-facing, the only sensor the neural network uses.

## AI Command Architecture (Phase 3 — deferred)

The next stage splits compute between a PC and the Pi over WiFi:

- **PC side** — receives the camera stream from the Pi, runs **YOLO** to build a "track memory" (objects seen, where and when). A user types a natural-language instruction; **Gemini** (LLM) receives the prompt + track memory and returns a structured JSON command like:
  ```json
  {"action": "go_to", "target": "parking", "on_reach": "stop"}
  ```
- **Pi side** — DonkeyCar keeps driving with the trained model, a small **state machine** watches for the target object, and when the target's bounding box grows past a threshold (i.e. we've arrived), it executes the `on_reach` action (stop, turn around, etc.).

This keeps the heavy AI/vision compute on the PC while the Pi only drives and reacts.

## Repository Structure

```
├── mycar/                  # DonkeyCar application (created with `donkey createcar`)
│   ├── manage.py           # Main vehicle loop — drive, record, autopilot
│   ├── ibus_receiver.py    # Custom DonkeyCar part: FlySky iBUS receiver over UART
│   ├── config.py           # DonkeyCar defaults
│   ├── myconfig.py         # Our overrides (camera type, drivetrain, phase flags)
│   ├── calibrate.py        # Servo/ESC PWM calibration tool
│   ├── train.py            # Phase 1: behavioral-cloning training entry point
│   ├── data/               # Recorded driving data (images + steering/throttle labels)
│   └── parts/              # Phase 2 only — dormant unless USE_CAMPUS_AUTONOMY = True
│       ├── seg_pilot.py        # Drivable-area segmentation + geometric steering
│       ├── yolo_guard.py       # Pretrained pedestrian/obstacle detection
│       ├── ultrasonic.py       # HC-SR04 x3 reflex stop layer
│       ├── breaker_detect.py   # Speed-breaker stripe detector (OpenCV, no ML)
│       ├── gps_nav.py          # Route following, junction commands, geofence
│       └── safety_arbiter.py   # Priority merge into final steering/throttle
├── scripts/                # Phase 2 tooling (run on a laptop, not the Pi)
│   ├── export_models.py        # Download + quantize the pretrained models
│   ├── build_campus_graph.py   # OpenStreetMap campus routing graph
│   └── vision_bench.py         # Offline go/no-go test on recorded footage
├── AUTONOMY.md             # Phase 2 architecture and setup
├── BUILD_STAGES.md         # Staged build guide: wiring, tests, parts list
├── lanedetection.py        # Classic OpenCV lane detection experiment (Canny + Hough lines)
├── test_camera.py          # Camera sanity check
└── test_server.py          # Simple HTTP server to verify Pi ↔ phone/PC networking
```

## Customizations on top of stock DonkeyCar

- **`ibus_receiver.py`** — a custom DonkeyCar *part* that reads the FlySky iBUS protocol directly over the Pi's UART (packet parsing, checksum validation, deadzone handling), instead of using the standard web/joystick controllers. Channel 5 on the transmitter switches between **manual** and **autopilot** mode.
- **PC-based development mode** — `myconfig.py` is set up with `CAMERA_TYPE = "CVCAM"` (USB webcam via OpenCV/DirectShow) and `DRIVE_TRAIN_TYPE = "MOCK"` so the full pipeline can be developed and tested on a Windows PC before deploying to the Pi.
- **Lane detection experiments** — `lanedetection.py` implements a classic computer-vision lane follower (grayscale → Gaussian blur → Canny edges → region-of-interest mask → Hough line transform → averaged lane lines) as a baseline/fallback to compare against the learned model.

## Getting Started

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install donkeycar[pc]     # on the Pi: donkeycar[pi]
```

### 2. Calibrate steering & throttle (on the Pi)

```bash
cd mycar
python calibrate.py
```

### 3. Collect training data

Drive manually with the FlySky transmitter; frames + control inputs are recorded to `mycar/data/`.

```bash
python manage.py drive
```

### 4. Train the model

```bash
python train.py --tubs data/ --model models/mypilot.h5
```

### 5. Let it drive

```bash
python manage.py drive --model models/mypilot.h5
```

Flip channel 5 on the transmitter to switch into autopilot.

> Steps 3–5 are **Phase 1**, the behavioral-cloning workflow, and are the current
> focus. Phase 2 (campus autonomy) is documented separately in
> [AUTONOMY.md](AUTONOMY.md) and stays completely out of the way until you set
> `USE_CAMPUS_AUTONOMY = True`.

## Acknowledgements

- [DonkeyCar](https://github.com/autorope/donkeycar) — the open-source DIY self-driving platform this project is built on
- NVIDIA's [End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) paper, the basis of the behavioral-cloning approach
