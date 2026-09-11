# Self-Driving RC Car — Campus Delivery Cart Prototype

A small autonomous vehicle built on [DonkeyCar](https://www.donkeycar.com/) and a
Raspberry Pi. The end goal is a **campus delivery cart**: it carries documents or
parcels between buildings with nobody driving it.

> **This is a prototype.** The point of this stage is to prove self-driving works
> end-to-end on cheap hardware before scaling it up.

## The three modes

The repo holds three separate ways of making the car see and steer. They do not
interfere with each other — modes 1 and 2 are picked by one flag in `myconfig.py`,
and mode 3 never runs on the car at all.

| | **1. DonkeyCar** | **2. Fast-SCNN + OpenCV** | **3. Mask2Former Teacher** |
|---|---|---|---|
| What it does | Copies how *you* drive | Real-time semantic road segmentation & steering | Offline teacher: auto-labels campus video for distillation |
| How | CNN trained on your recordings | Fast-SCNN INT8 mask → geometric corridor steering | 215M-param panoptic model on Cloud GPU |
| Training data | Human driving telemetry | Distilled from Mask2Former pseudo-labels | None — pretrained foundation model |
| Runs on the Pi | Yes (~20 FPS) | **Yes (7.11 FPS, 110 MB RAM)** | **No** — Cloud EC2 GPU training only |
| Turn it on | `USE_CAMPUS_AUTONOMY = False` *(default)* | `USE_CAMPUS_AUTONOMY = True` | `scripts/prepare_m2f_dataset.py` |
| Best for | Repeatable tracks, model comparison | Real-time campus navigation on 2GB Pi | Zero-manual-labeling dataset generation |
| Code | stock DonkeyCar + `ibus_receiver.py` | `mycar/parts/` + `pie/` | `scripts/` |

**Status:** Mode 2 (Fast-SCNN INT8) is deployed and benchmarked on real-world 64-bit Raspberry Pi 4 hardware at **7.11 FPS**. Mode 3 serves as the offline teacher model for automated dataset labeling.

---

### Mode 1 — DonkeyCar (behavioral cloning)

You drive the car manually while it records camera frames paired with your
steering and throttle. A CNN learns to imitate you. On the track, the Pi feeds it
live frames and it outputs controls ~20 times a second.

```
Drive manually → Record data → Train CNN → Car drives itself
```

The experiment: train across **5–6 different track layouts** and hold one out, to
measure whether the model *generalizes* or just *memorizes*. Per-track models are
compared against one combined model on lap success rate. The two ablations that
matter are `linear` vs `categorical` output, and `ROI_CROP_TOP` on vs off —
cropping the top of the frame removes track-specific background and is the single
biggest anti-overfitting lever.

Set `CREATE_TF_LITE = True`. Training runs on x86 and inference on ARM, and
mismatched TensorFlow versions are the most common "model won't load" failure.

### Mode 2 — Fast-SCNN + OpenCV Geometric Autonomy

A lightweight edge semantic segmentation network labels every pixel in real time. The drivable road pixels form an adaptive corridor, and steering is calculated geometrically over that corridor — fused with sensor arbitration:

- `seg_pilot.py` — Fast-SCNN drivable-corridor segmentation & lookahead steering (`roi_top = 0.30`)
- `yolo_guard.py` — Pretrained pedestrian and obstacle detection
- `ultrasonic.py` — HC-SR04 sonar reflex emergency stop layer
- `breaker_detect.py` — Speed-breaker strip detection via classical OpenCV
- `gps_nav.py` — Waypoint route following, junction turning, and geofence enforcement
- `safety_arbiter.py` — Priority arbitrator enforcing vehicle safety gates

### Mode 3 — Mask2Former Cloud Teacher & Knowledge Distillation

![Fast-SCNN semantic road segmentation on campus footage](docs/results/fastscnn_campus_demo.gif)

*Fast-SCNN (INT8 quantized, 1.7 MB) predicting drivable road boundaries in real time on recorded campus footage at `roi_top = 0.30`.*

A heavy 215M-parameter foundation model (Mask2Former Swin-L) is far too slow for embedded edge hardware (~7 FPS on an enterprise T4 GPU). However, it serves as the **offline teacher model** in our knowledge distillation and transfer learning pipeline.

---

## Knowledge Distillation & Transfer Learning Pipeline

Manual pixel-by-pixel annotation of campus video is prohibitively expensive. We developed an end-to-end cloud-to-edge distillation workflow:

```
[Raw Campus Video] 
       │
       ▼
[Cloud GPU: Mask2Former Swin-L (Teacher)] ──▶ [Auto-Generated Dense Road Masks]
                                                        │
                                                        ▼
                                         [Transfer Learning & Fine-Tuning]
                                         [Student: Fast-SCNN (~1.1M params)]
                                                        │
                                                        ▼
                                         [Validation: 0.9782 IoU / 0% Stops]
                                                        │
                                                        ▼
                                         [Dynamic INT8 Quantization (1.7 MB)]
                                                        │
                                                        ▼
                                         [Edge Deployment: Raspberry Pi 4B]
```

1. **Teacher Pseudo-Labeling (`scripts/prepare_m2f_dataset.py`):**
   Raw video frames (`selfDRIVING_cropped.mp4`, 1,302 frames) are processed on an AWS EC2 T4 GPU instance. Mask2Former extracts high-fidelity pseudo-ground-truth binary masks of the drivable road.
2. **Transfer Learning (`scripts/train_fastscnn.py --weights`):**
   The compact Fast-SCNN architecture is initialized from pretrained weights and fine-tuned on the distilled campus pseudo-labels using cross-entropy and Dice loss.
3. **Convergence & Zero Catastrophic Forgetting:**
   Training converged at epoch 100 with a validation **IoU of 0.9782**. Evaluated against existing campus footage (`campussample_trimmed.mp4`), the model maintained a **0% false emergency stop rate**, proving domain adaptation without losing past knowledge.
4. **INT8 Quantization (`scripts/export_fastscnn.py`):**
   The fine-tuned PyTorch checkpoint is exported to ONNX and quantized to dynamic INT8 (`fastscnn_selfdriving_int8.onnx`, 1.7 MB, 256×256 input).

---

## What the Campus Benchmarks Showed

### 1. Steering Lookahead Region Calibration (`roi_top = 0.30`)
The vertical start position of the steering evaluation window (`roi_top`) governs how far ahead the vehicle anticipates curves:

| `roi_top` Setting | Effective View | Steering Behavior | Result |
|---|---|---|---|
| `0.40` (Default) | Near-field bumper area | Stable on straights, but turns are initiated late | Sluggish response to sharp bends |
| `0.25` | Distant horizon view | Looks too far ahead; sensitive to background clutter | Twitchy reaction to distant curves |
| **`0.30` (Calibrated)** | **Mid-range lookahead** | **Smooth curve entry while staying locked to road** | **Optimal sweet spot (Locked)** |

### 2. The Ego-Vehicle Bumper Gotcha
If the camera sees the vehicle's own hood or handlebars, the model misidentifies vehicle plastic as drivable path or obstacle boundaries:
- **Raw footage (`selfDRIVING.mp4`):** Bottom 300px contained vehicle handlebars, corrupting lower centroid calculations.
- **Solution:** A 300px bottom crop (`--crop-bottom` / physical camera angle tilt) eliminated 100% of bodywork contamination before inference.

### 3. Bare-Metal Raspberry Pi 4 (2GB) Hardware Benchmark
Benchmarked directly on physical Raspberry Pi 4 hardware running **64-bit Debian Bookworm (`aarch64`)** in headless mode across 751 frames:

| Metric | Measured Value | Operational Significance |
|---|---|---|
| **Average Throughput** | **7.11 FPS** | Full road update every **136.6 ms** |
| **Temporal Jitter (p95)** | **144.9 ms** | 95% of frames complete within $\pm 8\text{ ms}$ of average |
| **Peak RAM Usage (RSS)** | **110.6 MB** | Consumes only **5.5%** of 2GB RAM (>1.88 GB free) |
| **Sustained Thermals** | **76.0 °C** | Safely below the 80–85 °C CPU throttling threshold |
| **Reaction Distance @ 10 km/h** | **0.38 m (1.2 ft)** | Cart reacts within one foot of travel distance |

## Hardware

```
FlySky Transmitter ──RF──▶ FlySky Receiver ──iBUS/UART──▶ Raspberry Pi ──I2C──▶ PCA9685 ──PWM──▶ Servo (steering)
                                                              │                              └───▶ ESC (throttle)
                                                          Pi Camera
```

- **FlySky TX/RX** — manual control, and channel 5 switches manual ↔ autopilot. It
  reaches the Pi over **iBUS** (all 14 channels on one digital wire, far cleaner
  than reading raw PWM).
- **Raspberry Pi** — runs the vehicle loop, reads iBUS, runs the model.
- **PCA9685** — I2C PWM driver. Direct GPIO PWM is too jittery for smooth steering.
- **Servo + ESC** — on an external battery rail; everything shares a common ground.
- **Camera** — front-facing, and the only sensor mode 1 uses.

## Repository structure

```
├── mycar/                  # DonkeyCar application
│   ├── manage.py           # Vehicle loop — drive, record, autopilot
│   ├── ibus_receiver.py    # Custom part: FlySky iBUS over UART
│   ├── myconfig.py         # Our overrides, including the mode flag
│   ├── calibrate.py        # Servo/ESC PWM calibration
│   ├── train.py            # Mode 1 training entry point
│   ├── data/               # Recorded frames + steering/throttle labels
│   └── parts/              # Autonomy stack (`seg_pilot.py` with `roi_top=0.30`)
├── pie/                    # Self-contained 64-bit Raspberry Pi deployment package
│   ├── test_pi.py          # Real-time hardware throughput & FPS benchmark
│   ├── stats.py            # Comprehensive telemetry suite (p95 latency, thermals)
│   ├── segment_video.py    # Real-time video generator with green road overlay
│   ├── bw_mask.py          # Binary black & white segmentation mask generator
│   ├── fastscnn_selfdriving_int8.onnx # Quantized 1.7 MB production model
│   └── fastscnn_labels.json # Category labels and input metadata
├── scripts/                # Cloud distillation & evaluation tooling
│   ├── prepare_m2f_dataset.py  # Mask2Former pseudo-labeling auto-extractor
│   ├── train_fastscnn.py       # Fast-SCNN transfer learning & fine-tuning
│   ├── export_fastscnn.py      # ONNX export and dynamic INT8 quantizer
│   ├── vision_bench.py         # Offline simulation benchmark with ROI tuning
│   └── stitch_campus.py        # Split-screen Before/After video comparator
├── memory/                 # Persistent unified project memory
│   └── PROJECT_MEMORY.md   # Single source of truth for decisions & telemetry
├── AUTONOMY.md             # Mode 2 architecture and setup
├── BUILD_STAGES.md         # Wiring, tests, parts list
└── lanedetection.py        # Classic OpenCV lane follower (Canny + Hough)
```

`exported_models/` is gitignored — it holds model binaries, so the per-profile
label files are generated rather than committed. Reproduce any profile above with
`python scripts/export_models.py --profile <name>`.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install donkeycar[pc]     # on the Pi: donkeycar[pi]
```

Then, on the Pi:

```bash
cd mycar
python calibrate.py
python manage.py drive
python train.py --tubs data/ --model models/mypilot.h5
python manage.py drive --model models/mypilot.h5
```

That is: calibrate steering and throttle, drive manually to record into `data/`,
train, then hand over. Flip channel 5 on the transmitter to switch into autopilot.

For mode 2, set `USE_CAMPUS_AUTONOMY = True` and see [AUTONOMY.md](AUTONOMY.md).
It stays completely out of the way until you do.

## Acknowledgements

- [DonkeyCar](https://github.com/autorope/donkeycar) — the platform this is built on
- NVIDIA's [End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), the basis of the behavioral-cloning approach
- Udacity's [self-driving car](https://github.com/udacity/self-driving-car) repo and [Challenge #2 writeup](https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c) — the winning entries are a useful reference for steering regression, particularly that framing it as classification first beats regressing angles directly
- [SegFormer](https://arxiv.org/abs/2105.15203) and [Mask2Former](https://arxiv.org/abs/2112.01527), and the [sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic) and [Cityscapes](https://www.cityscapes-dataset.com/) label sets used by the pretrained models
