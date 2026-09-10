# Unified Project Memory — Self-Driving Campus Delivery Cart

> **Purpose:** Single source of truth for the project. Structured so that any AI assistant (Claude, Antigravity, ChatGPT, Cursor, Copilot, Gemini) or collaborator can instantly understand the complete architecture, settled decisions, hardware constraints, and experimental protocols without rediscovering context.

---

## 1. Project Identity & Context

- **Project:** Small-scale autonomous campus delivery cart prototype (indoor & outdoor).
- **Repository:** [`https://github.com/GuptaOum/selfdriving-campus-cart`](https://github.com/GuptaOum/selfdriving-campus-cart) (GitHub account: `GuptaOum`).
- **Academic Context:** Final-year engineering project.
- **Competition Details (SIH 2026):**
  - **Problem Statement ID:** 26126 (*"Vision-Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment"*).
  - **Theme:** Robotics / AI / Autonomous Systems.
  - **Category:** **Software** (Sold as portable, site-agnostic autonomous software; the RC cart is the physical demonstrator).
  - **Team ID / Name:** `SIH26126` / **AUTOROVERS**.
  - **Pitch Rule:** Pitch hard and aggressively on capabilities; never fabricate benchmark metrics.

---

## 2. Core Working Principles & AI Assistant Guidelines

1. **Answer Short First:** Provide direct, concise answers first; elaborate or provide deep dives only when prompted.
2. **Zero Attribution in Repo:** No mentions of AI assistants, LLM co-authors, or tool artifacts in commits, docstrings, or code (`feedback-no-attribution-in-repo`).
3. **No Loss Ranking:** Never rank behavioral cloning models by training or validation loss (refuted by Codevilla ECCV 2018 & robomimic CoRL 2021). Closed-loop on-track performance is the only ground truth.
4. **Git Authentication:** Uses Git Credential Manager (`credential.helper=manager`); OAuth token works for git push and GitHub API calls (fetch via `git credential fill`). Never log or print auth tokens.

---

## 3. The Two-Phase Architecture

The codebase maintains a **strict architectural firewall** between two independent phases:

```
mycar/manage.py
 ├── Default: Phase 1 DonkeyCar Behavioral Cloning (USE_CAMPUS_AUTONOMY = False)
 └── add_campus_autonomy(): Phase 2 Advanced Pretrained Stack (USE_CAMPUS_AUTONOMY = True)
```

| Dimension | **Phase 1: DonkeyCar (Active Work)** | **Phase 2: Campus Autonomy (Dormant / Verified)** |
|---|---|---|
| **Core Method** | Behavioral cloning (human imitation via CNN) | Zero training data; pretrained vision + occupancy grid geometry |
| **Primary Goal** | Research contribution: Generalization vs. memorization ablation | Outdoor campus navigation, obstacle avoidance, GPS A-to-B |
| **Compute Location** | Model trained on Colab T4; inference on Raspberry Pi 4B | Runs directly on Pi 4B (TFLite / ONNX INT8 / NCNN) |
| **Status** | Active research focus | Built, verified (`tests/test_safety.py` ALL PASS), dormant |

---

## 4. Hardware Architecture & Pin Map

### Control & Signal Chain
```
FlySky TX --RF--> FlySky RX --iBUS/UART--> Raspberry Pi 4B (2GB) --I2C--> PCA9685
                                                                            |-- ch1 --> Steering Servo
                                                                            `-- ch0 --> ESC -> A2212 BLDC Motor
```

### Key Hardware Decisions (Settled — Do Not Re-litigate)
- **Raspberry Pi 4B (2 GB):** RAM is the hard binding constraint, not CPU.
- **PCA9685:** Dedicated I2C PWM driver gives jitter-free servo/ESC signals.
- **FlySky iBUS over UART (`/dev/serial0`, GPIO15):** Transmits all 14 channels cleanly over a single wire.
- **Master Kill-Switch / Override:** **FlySky CH5 3-position switch** (`IBUS_MODE_CHANNEL`). Priority #1 above all software, models, and sensors.
- **Power Isolation:** Separate battery rails with a **common ground**.
  - Motor & ESC: LiPo battery.
  - Raspberry Pi: Dedicated power bank delivering **5V / 3A** minimum (lower current browns out Pi under camera+WiFi load, mimicking software bugs). Check `vcgencmd get_throttled == 0x0`. Never power Pi from the ESC BEC.
- **Motor:** A2212 sensorless BLDC + ESC. (User confirmed torque is sufficient for the 4.5 kg chassis at low speed; plane ESCs have no reverse and need an arming sequence).
- **IMU Status:** **MPU6050 REMOVED.** (Causal confusion: past motion channels in imitation learning degrade driving policies; IMU off).
- **GPS:** Switched from NEO-6M to **u-blox M8N** connected via **USB-TTL** (since iBUS occupies `/dev/serial0`). Radii: `GPS_JUNCTION_RADIUS_M = 8.0`, `GPS_ARRIVE_RADIUS_M = 5.0`.
- **Sonars:** 4× HC-SR04 ultrasonic sensors.
  - **Critical:** Echo pins output 5V — **voltage dividers (1kΩ + 2kΩ) required** to step down to 3.3V GPIO.
  - **Mounting:** Mount $\ge 10\text{--}12\text{ cm}$ high, level, so standard $\approx 6\text{ cm}$ campus speed breakers do not trigger false emergency stops.
  - Sensor bearings: Left (+30°), Center-Left (+10°), Center-Right (-10°), Right (-30°).
- **Webcam:** Mounted at height $\approx 45\text{ cm}$ (~1.5 ft) looking ahead. Once calibrated, must not move. **Cart hood/bonnet must be cropped out of the frame** (see Vision Benchmarks).

### Raspberry Pi GPIO Map

| Function | BCM GPIO | Physical Header Pin |
|---|---|---|
| PCA9685 I2C (SDA / SCL) | GPIO2 / GPIO3 | Pins 3 / 5 |
| iBUS Input (UART RX) | GPIO15 (RXD0) | Pin 10 |
| Sonar Left (Trig / Echo) | GPIO5 / GPIO6 | Pins 29 / 31 |
| Sonar Center-Left (Trig / Echo) | GPIO19 / GPIO26 | Pins 35 / 37 |
| Sonar Center-Right (Trig / Echo) | GPIO20 / GPIO21 | Pins 38 / 40 |
| Sonar Right (Trig / Echo) | GPIO16 / GPIO12 | Pins 36 / 32 |
| GPS Receiver | — | USB-TTL port |
| USB Camera | — | USB 3.0 (blue port) |

---

## 5. Phase 1: DonkeyCar Behavioral Cloning Protocol

### Scientific Contribution & Novelty Position
- **Novelty:** Measuring the **sub-8 environment generalization boundary** on physical hobby hardware using closed-loop intervention metrics.
- **Refuted Claims (Avoid in papers/viva):** Sensor fusion (ALVINN 1988, KerasIMU), test-time modality failure (Liu 2017).
- **Unscoopable Contribution:** Re-deriving the **Bojarski takeover constant**. The 1968/2016 full-scale automotive constant assumes 6 seconds to regain control; on a 5m track, grab-and-recenter is 1–2 seconds. Stopwatch 20 real trials and report metrics under both constants.

### The Two Golden Protocol Rules
1. **Hold Total Frames Constant:** E.g., 12,000 frames total held across arms:
   $$\text{12k} \times 1 \quad\text{vs.}\quad 6\text{k} \times 2 \quad\text{vs.}\quad 4\text{k} \times 3 \quad\text{vs.}\quad 2\text{k} \times 6$$
   *Otherwise, diversity is confounded with dataset size.*
2. **$\ge 3$ Random Seeds Per Arm:** Single-run evaluations are invalidated by seed variance.

### Training Guidelines
- **Compute:** Free Google Colab (T4 GPU); images 160×120; 10k–20k frames take minutes. Set `CACHE_IMAGES = True`.
- **Model:** `KerasLinear` (PilotNet/DAVE-2 CNN: 5 conv layers → dense 100 → dense 50 → dual outputs) and `categorical` (often smoother on tracks).
- **TFLite Mandatory:** `CREATE_TF_LITE = True` in `myconfig.py` (x86 training vs. ARM inference compatibility).
- **Data Pipeline:** `ROI_CROP_TOP = 45`, `TRANSFORMATIONS = ['CROP']`, `AUGMENTATIONS = ['BRIGHTNESS', 'BLUR']`. Cropping background horizon is the #1 anti-overfitting mechanism.

---

## 6. Phase 2: Advanced Campus Autonomy Stack

### Architectural Philosophy: The Occupancy Grid
**No neural model directly outputs steering or throttle.** Every sensor writes into a 2D bird's-eye occupancy grid; geometry and planning algorithms compute actuation:

```
SegFormer-B0 ONNX    --> "Drivable corridor pixels"   ┐
YOLOv8n + ByteTrack  --> "Obstacles / Pedestrians"    ├─> Bird's-Eye Occupancy Grid (5-10 cm cells)
4× HC-SR04 Sonars    --> "Reflex distance buffers"    │        │
OpenCV Homography    --> Ground-plane transformation  ┘        ▼
                                                       LocalPlanner (21 arc rollouts)
                                                               │
                                                               ▼
                                                       SafetyArbiter (Clamps/Vetoes)
                                                               │
                                                               ▼
                                                       FlySky CH5 Hardware Override
```

### Models & Logic Components
1. **Drivable Segmentation (`seg_pilot.py`):**
   - Model: `segments-tobias/segformer-b0-finetuned-segments-sidewalk` (INT8 ONNX).
   - Class names carry `flat-` prefix (`flat-sidewalk`, `flat-crosswalk`, `flat-cyclinglane`).
   - Profile: Default is `footpath`. Excludes `flat-curb` and `nature-terrain`.
   - Path Geometry: Campus paths confirmed at **5 meters wide** (`PLANNER_LATERAL_M = 2.8`).
   - Junction Handling: `_target_x()` biases offset toward commanded edge while enforcing minimum vehicle clearance safety bounds.
2. **Obstacle Guard (`yolo_guard.py`):** Pretrained YOLOv8n (NCNN format) tracking dynamic pedestrians and campus hazards.
3. **Reflex Sonar Layer (`ultrasonic.py`):** Direct hardware interrupt distance cutoff.
4. **Speed Breaker Detector (`breaker_detect.py`):** Pure OpenCV zebra/stripe pattern detector.
5. **Local Planner (`local_planner.py`):** Dynamic Window arc evaluation; rolls out 21 bicycle-kinematic trajectories, penalizing obstacle collision, wrong-way heading, and jerkiness.
6. **Safety Arbiter (`safety_arbiter.py`):** Fails closed; enforces throttle limits and emergency stops.

### Ground Plane Calibration Rule
- Never rely on 4-point reprojection error (4 points fit a homography with 0.00 error mathematically, hiding human click mistakes).
- **Rule:** Always tape and measure a **5th verification marker** (`--verify X,Y`). Errors $<8\text{ cm}$ pass; $>15\text{ cm}$ require recalibration.

---

## 7. Vision Benchmarks & Empirical Findings

- **The Ego-Vehicle Bonnet Gotcha:**
  - If the bottom of the camera frame captures the cart's own bumper/hood, SegFormer's low-resolution receptive field labels the tarmac ahead as `vehicle-car` (drivable area collapsed from 52% to 8%, causing violent false steering into walls).
  - **Rule:** Physically aim camera up or apply bottom software crop before inference.
- **Resolution Scaling:** Higher resolution ($>320\times 240$) degrades real-time FPS on Raspberry Pi 4B without improving corridor centerline accuracy.
- **Steering Rate Limits:** `SEG_MAX_STEER_RATE = 1.2` dampens single-frame segmentation dropout spikes.

---

## 8. High-Level AI Command Architecture (PC + Pi Split)

For higher-level natural language control:
```
Pi camera → frames → PC
PC: YOLO builds a "track memory" (detected objects + timestamps/positions)
User types a natural-language prompt on PC
PC: LLM receives {prompt + track memory (text only, no images)} → structured JSON command
PC → Pi: command sent over WiFi socket
Pi: state machine — DonkeyCar drives normally, YOLO watches live;
    when target detected AND bounding-box area exceeds threshold → execute on_reach action (stop, do_360, etc.)
```
- **Why this split:** PC handles all heavy vision/reasoning compute; Pi runs the high-frequency deterministic drive loop.

---

## 9. Verification & Test Commands

Before committing or deploying changes to autonomy:
```bash
# Compile check all scripts and parts
python -m py_compile mycar/parts/*.py mycar/manage.py mycar/myconfig.py scripts/*.py tests/*.py

# Run comprehensive safety test suite (100+ assertions, mock hardware)
python tests/test_safety.py
```
*`test_safety.py` must print `ALL PASS`.*

---

## 10. Key File Index

---

## 11. Perception Model Fine-Tuning & Hardware Benchmarking (Fast-SCNN INT8)

### Transfer Learning Protocol & Pseudo-Labeling
- **Dataset Generation (`scripts/prepare_m2f_dataset.py`):** Used pretrained Mask2Former on EC2 GPU (`selfdriving-seg-lab`, `i-01127ed4e3d46b15e`) to extract ground-truth masks for `selfDRIVING_cropped.mp4` (1,302 frames, 300px bottom crop, 550px height).
- **Fine-Tuning (`scripts/train_fastscnn.py`):** Initialized from `fastscnn_campus_m2f_best.pth`. Early-stopped at epoch 100 with validation IoU **0.9782**.
- **Quantization (`scripts/export_fastscnn.py`):** Exported to dynamic INT8 ONNX (`fastscnn_selfdriving_int8.onnx`, 1.7 MB, input 256×256).
- **Catastrophic Forgetting Gate:** Verified on original campus sample (`campussample_trimmed.mp4` & `campussample_middle.mp4`): maintained **0% false emergency stop rate** across all 2,000+ benchmark frames.

### Settled Steering ROI Height (`roi_top = 0.30`)
- Evaluated empirical lookahead boundaries:
  - `0.40`: Original default (too near-field, delayed reaction to sharp turns).
  - `0.25`: Too high (over-sensitive to horizon and distant background clutter).
  - **`roi_top = 0.30` (Locked Decision):** Perfect compromise; anticipates turns smoothly while staying focused on immediate drivable road. Configured in [`mycar/parts/seg_pilot.py`](file:///d:/selfdriving/mycar/parts/seg_pilot.py).

### Raspberry Pi 4 (2GB) Hardware Benchmark (64-Bit OS)
- **OS Architecture Decision:** Upgraded from 32-bit `armhf` to **pure 64-bit Debian Bookworm (`aarch64`)**. Pi 4 boots natively from 64GB USB 3.0 drive; desktop display manager disabled (`multi-user.target`) for headless efficiency.
- **Bare-Metal Telemetry (`pie/stats.py` on 751 frames, 4 CPU threads):**
  - **Average Throughput:** **7.11 – 7.18 FPS** (Average latency: **136.6 ms / frame**).
  - **Temporal Consistency (Jitter):** 95th-percentile (p95) latency **144.9 ms** ($\pm 8\text{ ms}$ variance across whole run).
  - **Memory Footprint:** **110.6 MB RSS** (only 5.5% of the 2,000 MB RAM budget; leaves $>1.88\text{ GB}$ free).
  - **Thermal Performance:** Peaked at 76.0 °C under sustained 100% 4-core CPU load without thermal throttling.
  - **Vehicle Reaction Distance:** At 10 km/h (~2.8 m/s), path updates every **0.38 m (1.2 ft)**.

### Remote Access & Field Telemetry Infrastructure
- **Local mDNS:** `ssh oum@oum.local` connects across direct Ethernet cable or local Wi-Fi without knowing IP address.
- **Global Mesh (Tailscale):** Static permanent IP **`100.80.56.92`** (`tailscale0`). Connects securely across mobile hotspot, college Wi-Fi, or remote networks with zero port forwarding.
- **Collaborator Access:** Configured via Tailscale Node Sharing (inviting external accounts without sharing credentials).

---

## 12. Key File Index

- [`PROJECT_MEMORY.md`](file:///d:/selfdriving/memory/PROJECT_MEMORY.md) — This document (Unified Project Memory).
- [`README.md`](file:///d:/selfdriving/README.md) — Public-facing architecture and quickstart guide.
- [`AUTONOMY.md`](file:///d:/selfdriving/AUTONOMY.md) — Comprehensive guide to Phase 2 campus autonomy.
- [`BUILD_STAGES.md`](file:///d:/selfdriving/BUILD_STAGES.md) — Physical build, electrical wiring gates, and bring-up stages.
- [`mycar/myconfig.py`](file:///d:/selfdriving/mycar/myconfig.py) — Vehicle configuration overrides and autonomy flags.
- [`mycar/manage.py`](file:///d:/selfdriving/mycar/manage.py) — DonkeyCar main drive loop + campus autonomy injector.
- [`mycar/parts/seg_pilot.py`](file:///d:/selfdriving/mycar/parts/seg_pilot.py) — Semantic segmentation steering pilot (`roi_top = 0.30`).
- [`scripts/train_fastscnn.py`](file:///d:/selfdriving/scripts/train_fastscnn.py) — Fast-SCNN PyTorch training & transfer learning.
- [`scripts/export_fastscnn.py`](file:///d:/selfdriving/scripts/export_fastscnn.py) — Fast-SCNN ONNX FP32 & INT8 quantizer.
- [`scripts/prepare_m2f_dataset.py`](file:///d:/selfdriving/scripts/prepare_m2f_dataset.py) — Mask2Former auto-segmentation labeler for videos.
- [`scripts/vision_bench.py`](file:///d:/selfdriving/scripts/vision_bench.py) — Offline vision benchmark tool on recorded clips.
- [`scripts/fastscnn_bw_mask.py`](file:///d:/selfdriving/scripts/fastscnn_bw_mask.py) — Binary black & white drivable mask generator.
- [`scripts/stitch.py`](file:///d:/selfdriving/scripts/stitch.py) / [`stitch_campus.py`](file:///d:/selfdriving/scripts/stitch_campus.py) — Split-screen Before/After evaluation tools.
- [`pie/`](file:///d:/selfdriving/pie/) — Self-contained Raspberry Pi deployment directory:
  - `test_pi.py` — Quick throughput/FPS verification.
  - `stats.py` — Full telemetry reporting suite (FPS, p95 latency, thermals, RAM).
  - `segment_video.py` — Video generator with transparent green road overlay.
  - `bw_mask.py` — Binary black & white mask generator.
  - `fastscnn_selfdriving_int8.onnx` — Production fine-tuned INT8 model (1.7 MB).
  - `fastscnn_labels.json` — Label mapping and metadata.
  - `seg_pilot.py` — Synchronized DonkeyCar steering logic.
- [`tests/test_safety.py`](file:///d:/selfdriving/tests/test_safety.py) — Offline unit & safety assertion tests.

