# Campus Autonomy Stack — Pretrained Vision (No Training Data)

> **This is Phase 2 and it is dormant by default.** Phase 1 — behavioral cloning on
> indoor tracks — remains the current focus and is entirely unaffected by anything
> here. With `USE_CAMPUS_AUTONOMY = False` (the default) none of these parts import
> or run, and `manage.py` behaves exactly like stock DonkeyCar. See the README for
> the phase plan.

The cart drives itself with **pretrained models only**: a SegFormer-B0 sidewalk
segmentation model finds the drivable path, geometric steering follows it, a
COCO YOLO watches for pedestrians, HC-SR04s provide a no-ML reflex stop, and
GPS supplies map position + junction commands (never steering).

```
                         ┌──────────────────────────────────────────────┐
 FlySky RC (CH5) ──────► │ DriveMode: 'user' mode ignores pilot/* ──────┼─► PCA9685
                         └──────────────▲───────────────────────────────┘   servo+ESC
                                        │ pilot/angle, pilot/throttle
                              ┌─────────┴──────────┐
                              │   SafetyArbiter    │  priority:
                              └─────────▲──────────┘  1 sonar stop <30cm
   HC-SR04 ×3 ── sonar/stop,bias ───────┤             2 gps unsafe (fail closed)
   GpsNav ────── nav/safe,command ──────┤             3 yolo person stop
   YoloGuard ─── yolo/stop,slow ────────┤             4 corridor blocked
   BreakerDetect breaker/active ────────┤             5 breaker creep mode
   SegPilot ──── seg/angle,throttle ────┘             6 drive (seg + sonar bias)
```

## Everything fails closed

The dangerous failure for an unattended vehicle is not a crash — a crashed part
is obvious. It is a part that **keeps returning its last good value** after it
has stopped working, so the cart drives confidently on stale information. Each
layer therefore detects its own silence and reports the unsafe answer:

| Failure | Detected by | Result |
|---|---|---|
| Camera freezes (same frame repeats) | `SegPilot` frame-identity + age check | corridor blocked → stop |
| Segmentation thread stalls or dies | `SegPilot` result age > `SEG_MAX_RESULT_AGE` | corridor blocked → stop |
| Detector throws repeatedly | `YoloGuard` consecutive-failure count | reports "person ahead" → stop |
| Detector thread stalls | `YoloGuard` result age | reports "person ahead" → stop |
| HC-SR04 unplugged / wire off | no rising edge on ECHO at all | `sonar/healthy` False → stop |
| GPS lost, stale, or outside fence | fix age + polygon test | `nav/safe` False → stop |
| Any part raises | caught, backs off 200 ms, holds the stop | stop |

The sonar case is the subtle one. A **working** HC-SR04 with clear air ahead
still raises ECHO — it emits a ~38 ms "nothing there" pulse. A **disconnected**
one never raises ECHO at all. Reading silence as "path clear" would let the cart
drive with no proximity sensing whatsoever, so `read_cm()` separates the two and
the array refuses to report clear when a sensor has gone quiet.

Health flags are only enforced for layers you actually enabled — `require_sonar`
and `require_yolo` are wired from `HAVE_ULTRASONIC` / `HAVE_YOLO_GUARD`, so a
layer you never installed is not reported as a broken one.

Run `python tests/test_safety.py` after touching any of this. It needs no
hardware, no models and no Pi, and it asserts each row of the table above.

## Files

| File | What it is |
|---|---|
| `mycar/parts/seg_pilot.py` | SegFormer ONNX inference + band-centroid steering |
| `mycar/parts/yolo_guard.py` | YOLOv8n NCNN pedestrian/obstacle guard |
| `mycar/parts/ultrasonic.py` | pigpio HC-SR04 ×3, median filter, tiered stop |
| `mycar/parts/breaker_detect.py` | classical-CV yellow/black stripe detector |
| `mycar/parts/gps_nav.py` | gpsd + graphml routing, junction commands, geofence |
| `mycar/parts/safety_arbiter.py` | priority merge → `pilot/angle`, `pilot/throttle` |
| `scripts/export_models.py` | laptop/Colab: download + ONNX/INT8/NCNN export |
| `scripts/build_campus_graph.py` | laptop: OSM campus graph → graphml |
| `scripts/vision_bench.py` | Phase 1 go/no-go on recorded footage |
| `tests/test_safety.py` | Steering geometry + fail-closed tests (no hardware) |

## Setup — laptop (once)

```bash
pip install transformers torch onnx onnxruntime ultralytics osmnx
python scripts/export_models.py
python scripts/build_campus_graph.py --point <lat>,<lon> --dist 800
```

Copy `exported_models/` into `mycar/models/` on the Pi and
`campus_graph.graphml` into `mycar/`.

## Setup — Pi 4B 2GB (mandatory, RAM is the constraint)

- Raspberry Pi OS **Lite 64-bit**, headless. Enable zram, not SD swap.
- `sudo apt install pigpiod gpsd && sudo systemctl enable --now pigpiod`
- venv: `pip install onnxruntime ultralytics networkx` (+ existing donkeycar)
- One process only — everything runs inside `manage.py`'s loop.
- Check power: `vcgencmd get_throttled` must be `0x0`.

## Phase 1 go/no-go (BEFORE the car moves)

Record clips walking the routes (include sandy stretches + a speed breaker):

```bash
python scripts/vision_bench.py --video campus_walk.mp4 \
  --seg-model models/exported_models/segformer_sidewalk_int8.onnx \
  --seg-labels models/exported_models/segformer_labels.json \
  --yolo-model models/exported_models/yolov8n_ncnn_model --out annotated.mp4
```

- Mask bad on sand/dirt → fine-tune with ~100 labeled campus images on Colab
  (https://huggingface.co/blog/fine-tune-segformer), re-export, re-bench.
- OOM or seg < 1 FPS → upgrade to Pi 5 8GB (code unchanged).
- Both fine → enable flags in `myconfig.py` (bottom section) and bench-test
  with wheels off the ground: RC override, sonar stop, then track test.

## Hardware notes

- **HC-SR04s ≥ 10–12 cm high, aimed level** — otherwise every ~6 cm speed
  breaker trips the 30 cm hard stop forever.
- Speed breakers: cross perpendicular at creep; physical gate = wheel radius ≥
  breaker height and belly clearance at full compression. Routes that fail are
  marked impassable in the graph, not driven.
- Calibrate `SEG_CORRIDOR_FRAC` after taping a 1 m grid and measuring how many
  pixels the (bot width + margin) corridor spans at the bottom image band.
