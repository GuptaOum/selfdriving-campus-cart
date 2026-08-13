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
| `mycar/parts/gps_nav.py` | gpsd, route following, junction commands, geofence |
| `mycar/parts/mission_client.py` | polls EC2 for a route, reports position |
| `mycar/parts/safety_arbiter.py` | priority merge → `pilot/angle`, `pilot/throttle` |
| `server/app.py` | EC2: campus routing, mission state, operator API |
| `server/static/index.html` | the phone app — drop a pin, watch the cart |
| `scripts/export_models.py` | laptop/Colab: download + ONNX/INT8/NCNN export |
| `scripts/build_campus_graph.py` | laptop: OSM campus graph → graphml |
| `scripts/vision_bench.py` | Phase 1 go/no-go on recorded footage |
| `tests/test_safety.py` | Steering geometry + fail-closed tests (no hardware) |

## Deploying to the Pi

**Deploy early, not once at the end.** Stage 2 — the go/no-go that decides
whether this approach works at all — needs only the Pi, a webcam and the
models. No GPS, no sonar, no wiring. Get code onto the Pi as soon as it boots;
waiting until every part is soldered means discovering a model or RAM problem
after you have already spent the money.

**Code** comes over with git:

```bash
git clone https://github.com/GuptaOum/selfdriving-campus-cart.git
cd selfdriving-campus-cart
python -m venv .venv && source .venv/bin/activate
pip install donkeycar[pi]
pip install -r requirements-pi.txt
```

**Models do NOT come over with git.** They are gitignored on purpose — they are
large binaries and git is the wrong tool for them. Copy them separately, from
the laptop where you ran `export_models.py`:

```bash
scp -r exported_models pi@raspberrypi.local:~/selfdriving-campus-cart/mycar/models/
```

The campus graphml goes to **EC2, not the Pi** — routing happens on the server
now, so the Pi needs neither the graph nor networkx.

Same for anything else gitignored: recorded `.mp4` footage, trained `.h5`
models, and tub data all move by `scp`, never by `git push`.

Then confirm the paths line up with `myconfig.py`:

```
mycar/models/exported_models/segformer_sidewalk_int8.onnx
mycar/models/exported_models/segformer_labels.json
mycar/models/exported_models/yolov8n_ncnn_model/
```

**Sanity-check before trusting anything**, on the Pi:

```bash
python tests/test_safety.py          # no hardware needed; must print ALL PASS
vcgencmd get_throttled               # must be 0x0
free -m                              # confirm headroom on 2 GB
```

To update later, `git pull` on the Pi. Only re-`scp` models when you have
re-exported them.

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
- venv: `pip install -r requirements-pi.txt` (+ `pip install donkeycar[pi]`)
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

## Mission server (EC2) and the split that matters

The phone app and the campus routing live on an EC2 box; the cart polls it.
`server/app.py` is the whole server, `server/static/index.html` the whole app.

**What goes where, and why:**

| EC2 | Raspberry Pi |
|---|---|
| campus graph + networkx routing | steering, throttle |
| the operator's map, mission state | obstacle stops, corridor test |
| position history | **the geofence** |
| latency-tolerant | everything with a deadline |

**Nothing in the control loop crosses the network.** Campus WiFi to Mumbai and
back is 50-200 ms on a good day and unbounded on a bad one — fine for "go to
this pin", catastrophic for "stop". So a mission is handed over as a complete
waypoint list, and the cart drives it alone. If the server dies mid-delivery
the cart still finishes, because the route is already local. Losing a server
must not strand a cart in a corridor.

The geofence stays on the Pi for the same reason: it is fail-closed safety, and
safety that depends on a reachable server is not safety.

**The cart polls out.** Campus NAT blocks inbound connections, so an outbound
poll is the only thing that works without asking IT for a port forward. It also
means the Pi needs no public address at all.

### Running it

```bash
# on EC2 — t4g.small is plenty
pip install -r server/requirements.txt
export CART_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
python server/app.py --graph campus_graph.graphml
```

Copy the graphml from `scripts/build_campus_graph.py` to the server, **not** to
the Pi. On the Pi set `HAVE_MISSION_CLIENT = True`, `MISSION_SERVER_URL`, and
`MISSION_TOKEN` to the same value as `CART_TOKEN`.

The token is the only thing standing between the open internet and a moving
vehicle. Put nginx with TLS in front before this is anything but a prototype,
and keep the security group narrow.

## What actually fits on a bare Pi 4B 2GB (no upgrade)

You can have a genuinely autonomous A→B campus cart on this hardware. The one
thing to give up is YOLO.

**Why YOLO is the thing to cut.** `ultralytics` imports PyTorch at load time,
costing several hundred MB of RSS before it sees a frame — on a 2 GB box that
is most of your headroom, spent on the layer you need least. Segmentation
already labels a person `human-person`, i.e. not drivable, so **a person is
already a hole in the mask**, and the LocalPlanner already steers around holes
in the mask. YOLO only adds early warning at range. Losing it means you react
to a pedestrian at ~3 m from the mask instead of ~10 m from a detection, which
slow speed covers.

```python
USE_CAMPUS_AUTONOMY = True
HAVE_YOLO_GUARD    = False   # the RAM decision
HAVE_ULTRASONIC    = True    # cheap, and now your primary close-range safety
HAVE_LOCAL_PLANNER = True    # ~5 ms; this is what avoids obstacles
HAVE_GPS_NAV       = True    # negligible cost
HAVE_MISSION_CLIENT = True   # urllib only, no dependency
HAVE_BREAKER_DETECT = True   # ~2 ms of OpenCV

SEG_THROTTLE_CRUISE = 0.20   # see the speed maths below
SEG_THROTTLE_CREEP  = 0.14
```

**Budget, roughly:** Python + OpenCV + DonkeyCar ~250 MB, SegFormer INT8 at
256 px ~250 MB, camera buffers ~50 MB. Comfortable inside 1.6 GB. Add
ultralytics and you are fighting the OOM killer.

**Speed is set by inference rate, not by the motor.** At ~1.5 FPS the cart gets
one decision every 0.67 s. At 0.5 m/s it travels 33 cm between decisions, which
a 3 m planning horizon and a 30 cm sonar stop absorb comfortably. At 1 m/s it is
67 cm and the margin is gone. **Cap it near 0.5 m/s** — slow walking pace, which
is what a campus delivery cart should be doing anyway.

**If you want more speed, change the model before changing the board.**
SegFormer is a transformer and its attention does not vectorise well on ARM
CPUs. A CNN of similar accuracy — **Fast-SCNN** or a MobileNet-backed
segmentation head, both available pretrained on Cityscapes, which has `road`
and `sidewalk` — typically runs several times faster on the same silicon.
Fast-SCNN is what the USF campus sidewalk robot used. Swapping it in means a
different export in `export_models.py` and a different label mapping; nothing
else in the stack changes.

**The honest ceiling on this hardware:** follows paths, steers around people
and obstacles, stops for anything close, crosses speed breakers, navigates
A→B by GPS, reports to the app. What it will not do is react to a fast-moving
obstacle, drive faster than a slow walk, or see potholes.
