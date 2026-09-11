# Self-Driving RC Car — Campus Delivery Cart Prototype

A small autonomous vehicle built on [DonkeyCar](https://www.donkeycar.com/) and a
Raspberry Pi. The end goal is a **campus delivery cart**: it carries documents or
parcels between buildings with nobody driving it.

> **This is a prototype.** The point of this stage is to prove self-driving works
> end-to-end on cheap hardware before scaling it up.

<p align="center">
  <img src="docs/results/fastscnn_campus_demo.gif" width="48%" alt="Live Campus Segmentation"/>
  <img src="docs/results/bw_mask_output.gif" width="48%" alt="Real-Time B&W Road Mask"/>
  <br/>
  <em><b>Dual-View Real-Time Edge Inference (~7.5 FPS on Raspberry Pi 4B):</b> Left: Camera view with Fast-SCNN drivable corridor overlay. Right: Extracted Black & White binary road mask (<a href="docs/results/bw_mask_output.mp4"><b>bw_mask_output.mp4</b></a>) fed into the geometric planner.</em>
</p>

---

## Perception, Steering & GPS Architecture

Navigating narrow pedestrian walkways between campus buildings requires solving a classic robotics challenge: **civilian GPS alone cannot steer a 30 cm cart on a 1.5 m footpath (civilian GPS has a 3–5 m error margin).**

The system achieves autonomy by combining macro-routing with a dynamic micro-corridor path planner:

- **Dynamic Vision-Based Path Planning (Micro):** For every single camera frame, we first segment the drivable portion of the road while actively masking out non-navigable areas and obstacles like pedestrians or parked cars. We use our own custom-trained Fast-SCNN model—fine-tuned specifically on campus data because it yields vastly superior results to off-the-shelf models—running directly on the Raspberry Pi 4B CPU.
- **Geometric Steering (NumPy & OpenCV):** From that clean road mask, OpenCV and NumPy immediately calculate the nominal safe path to follow for that exact frame. By extracting spatial moments across horizontal bands, we find the path's center of mass to compute lateral error and heading angle (Δx, Δθ). Repeating this calculation continuously (~7.5 FPS) creates a **dynamic path planner** that actively adapts the cart's trajectory to the changing real-time environment.
- **GPS Waypoints & Junction Bias (Macro):** `gps_nav.py` follows an OSMnx campus graph between buildings. At pathway forks and intersections, GPS injects a directional bias (`junction_bias`), pulling the vision corridor aim point toward the intended branch. A fail-closed geofence halts the cart if GPS fix is lost or boundaries are crossed.

---

## Edge Transfer Learning Comparison

<p align="center">
  <img src="docs/results/side_by_side_comparison.gif" width="600" alt="Pretrained vs Fine-tuned Fast-SCNN"/>
  <br/>
  <em><b>Pre-Trained (Left) vs Fine-Tuned (Right):</b> Domain-adapted Fast-SCNN eliminates road dropouts, suppresses background bleeding, and cleanly tracks path boundaries. At 8–10 km/h the cart gets a new steering decision every 33–38 cm of travel.</em>
</p>

---

## Knowledge Distillation & Transfer Learning

Training dense segmentation on custom campus roads without manual labeling:

```
[Raw Campus Video] ──▶ [Cloud Teacher: Mask2Former Swin-L] ──▶ [Dense Road Pseudo-Labels]
                                                                        │
[Edge Deployment: Raspberry Pi 4B] ◀── [INT8 Quantization] ◀── [Student: Fast-SCNN Fine-Tuning]
       (~7.5 FPS, 110 MB RAM)              (1.7 MB ONNX)              (Val IoU: 0.9782)
```

1. **Teacher Pseudo-Labeling:** 215M-parameter Mask2Former on AWS EC2 T4 GPU auto-labels 1,300+ campus video frames.
2. **Student Fine-Tuning:** Lightweight Fast-SCNN (~1.1M params) trained on distilled labels, achieving **0.9782 Val IoU** and **0% false emergency stops**.
3. **INT8 Quantization:** Exported to dynamic INT8 ONNX (`1.7 MB`), requiring only **110 MB RAM** on the Pi.

---

## Hardware Benchmarks (Raspberry Pi 4B, 64-bit OS)

Measured across 751 frames running headless on Debian Bookworm (`aarch64`):

| Metric | Measured Value | Practical Significance |
|---|---|---|
| **Inference Throughput** | **~7.5 FPS** (136.6 ms) | Real-time steering updates |
| **Model Size** | **1.7 MB** (INT8 ONNX) | Ultra-lightweight edge footprint |
| **Memory Usage (RSS)** | **110.6 MB** | Only 5.5% of 2GB RAM; zero OOM risk |
| **Sustained Thermals** | **76.0 °C** | Safe margin below 80 °C CPU throttle point |
| **Lookahead Horizon** | **`roi_top = 0.30`** | Mid-range lookahead preventing curve lag |

---

## Autonomy Modes

| Mode | Technology | Role | Runs on Pi? |
|---|---|---|:---:|
| **1. DonkeyCar** | Behavioral Cloning (CNN) | Track imitation from human demonstrations | Yes (~20 FPS) |
| **2. Fast-SCNN + OpenCV** | Geometric Segmentation | Real-time corridor segmentation & lookahead steering | **Yes (~7.5 FPS)** |
| **3. Mask2Former** | 215M Foundation Model | Offline teacher for zero-annotation dataset creation | No (Cloud GPU) |

---

## Hardware Architecture

```
FlySky Transmitter ──RF──▶ FlySky Receiver ──iBUS/UART──▶ Raspberry Pi 4B ──I2C──▶ PCA9685 ──PWM──▶ Steering Servo
                                                                │                                    └──▶ Throttle ESC
                                                            Pi Camera
```

- **Raspberry Pi 4B (2GB):** Runs edge ONNX runtime, iBUS decoder, and vehicle safety arbiter.
- **PCA9685:** 16-channel 12-bit I2C PWM driver for stable servo/ESC pulses.
- **FlySky TX/RX:** Manual override; Ch 5 switch flips instantly between Manual and Autonomous.

---

## Engineering Behind the Scenes

### Why Vision, Not LiDAR or Lane Lines?

Campus footpaths have no lane markings, no kerb reflectors, and no standardised width. Classical Hough-transform lane detection fails immediately. LiDAR solves geometry but costs more than the entire cart. Semantic segmentation is the only sensor modality that can distinguish "walkable concrete" from "grass border" from "brick planter" using a single \$15 camera module — and Fast-SCNN is small enough to run on the Pi's CPU without a GPU or accelerator.

### The Calibration Decisions That Actually Mattered

- **`roi_top = 0.30`** — the vertical fraction of the frame where steering evaluation begins. Too low (`0.40`) and the cart steers late into curves. Too high (`0.25`) and distant horizon clutter causes twitchy corrections. `0.30` was found empirically by running the offline benchmark (`scripts/vision_bench.py`) across the full campus video and comparing false-stop rates and steering smoothness.
- **Bottom crop (300px)** — the cart's own handlebars appeared in the camera frame. The segmentation model's wide receptive field bled the `vehicle-car` label upward over the road, collapsing drivable area to 8% and steering the cart off-path. Cropping the bottom 16% of the frame (or tilting the camera) eliminated 100% of self-occlusion artifacts.
- **9px morphological closing** — campus paths include brick pavers with grass joints, concrete grids, and dappled tree shadows. Without morphological post-processing, the segmentation mask fragments into dozens of tiny disconnected regions that the corridor planner rejects as too narrow. A 9px closing kernel bridges these gaps while preserving real path boundaries.
- **PD gains (`kp=1.2, kd=0.3`)** — pure proportional control oscillates on straight paths. The derivative term damps heading corrections so the cart tracks the corridor centerline without weaving. These gains were tuned on physical hardware, not simulation.

### How the Safety Stack Works

The safety arbiter (`safety_arbiter.py`) is a deliberate, readable priority if-chain — not a learned policy. Priority order (highest first):

1. **RC transmitter override** — FlySky Ch 5 instantly hands control back to the human operator, bypassing all autonomy.
2. **HC-SR04 sonar hard stop** — ultrasonic reflex braking at < 30 cm, independent of vision processing latency.
3. **Geofence violation** — no GPS fix or position outside campus boundary polygon → fail-closed stop.
4. **YOLO pedestrian detection** — person detected in the corridor → stop or throttle reduction.
5. **Segmentation corridor blocked** — no drivable pixels found → stop.
6. **Normal operation** — segmentation steering angle + cruise throttle.

The arbiter is intentionally simple because this is the code a safety reviewer (or you at 2 AM after a crash) needs to audit at a glance.

See [docs/AUTONOMY.md](docs/AUTONOMY.md) for full perception configuration and [memory/PROJECT_MEMORY.md](memory/PROJECT_MEMORY.md) for calibration history and telemetry logs.

## Acknowledgements

- [DonkeyCar](https://github.com/autorope/donkeycar) — autonomous vehicle platform
- [Fast-SCNN](https://arxiv.org/abs/1902.04502) — Fast Semantic Segmentation Network
- [Mask2Former](https://arxiv.org/abs/2112.01527) — Universal Image Segmentation
- [Cityscapes](https://www.cityscapes-dataset.com/) & [Sidewalk-Semantic](https://huggingface.co/datasets/segments/sidewalk-semantic) — benchmark datasets
