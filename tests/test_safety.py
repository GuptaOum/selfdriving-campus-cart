"""
Safety and logic tests for the Phase 2 autonomy parts.

Runs on any machine — no Pi, no camera, no models, no pigpio. Model-loading
and GPIO constructors are bypassed so the pure logic is exercised directly.

    python tests/test_safety.py

Two things are being proved here:

  1. The steering geometry does what it claims (direction, corridor width,
     junction branch selection, throttle vs corridor depth).
  2. Every layer fails CLOSED. A dead sensor, a frozen camera, a stalled
     inference thread, or a crashed detector must all produce "stop", never
     "all clear". This is the property that makes the cart safe to leave
     driving on its own, and it is easy to regress by accident.
"""
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mycar"))

from parts.seg_pilot import SegEngine, SegPilot, _runs, _target_x     # noqa: E402
from parts.yolo_guard import YoloGuard                                # noqa: E402
from parts.ultrasonic import UltrasonicArray, NOTHING_IN_RANGE, NO_RESPONSE  # noqa: E402
from parts.breaker_detect import BreakerDetect, detect_breaker        # noqa: E402
from parts.gps_nav import (GpsNav, point_in_polygon, bearing_deg,     # noqa: E402
                           haversine_m)
from parts.safety_arbiter import SafetyArbiter                        # noqa: E402
from parts.local_planner import LocalPlanner                          # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAILURES.append(name)


def section(title):
    print(f"\n--- {title} ---")


def make_engine(mask_close_px=9, junction_bias=0.7, max_steer_rate=0.0):
    """A SegEngine with the steering parameters but no ONNX session."""
    e = SegEngine.__new__(SegEngine)
    e.kp, e.kd = 1.2, 0.0
    e.bands, e.roi_top = 5, 0.4
    e.corridor_frac_bottom = 0.28
    e.junction_bias = junction_bias
    e.max_steer_rate = max_steer_rate     # off by default so the geometry
    e._prev_angle = 0.0                   # tests measure geometry, not slew
    e.throttle_cruise, e.throttle_creep = 0.30, 0.16
    e._prev_offset, e._prev_time, e._prev_centroid_frac = 0.0, None, 0.5
    e.mask_close_px = mask_close_px
    e._close_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (mask_close_px, mask_close_px))
        if mask_close_px and mask_close_px > 1 else None)
    return e


def grass_paver_mask(size, x0, x1, cell=14, hole=8):
    """
    A concrete grid with grass growing through it: drivable everywhere inside
    [x0, x1) EXCEPT a regular lattice of holes. This is what a turf-block path
    actually segments to, and it is the pattern that used to defeat the
    corridor test.
    """
    m = np.zeros((size, size), np.uint8)
    m[:, x0:x1] = 1
    for y in range(0, size, cell):
        for x in range(x0, x1, cell):
            m[y:y + hole, x:x + hole] = 0
    return m


# =====================================================================
section("steering geometry")
# =====================================================================
S = 256
eng = make_engine()

mask = np.zeros((S, S), np.uint8); mask[:, 88:168] = 1
a, t, clear, dbg = eng.steer_from_mask(mask)
check("straight corridor centered", abs(a) < 0.05 and clear and t > 0.25,
      f"a={a:.2f} t={t:.2f}")

eng = make_engine()
mask = np.zeros((S, S), np.uint8); mask[:, 150:250] = 1
a, t, clear, _ = eng.steer_from_mask(mask)
check("right corridor steers right", a > 0.3 and clear, f"a={a:.2f}")

eng = make_engine()
a, t, clear, _ = eng.steer_from_mask(np.zeros((S, S), np.uint8))
check("blocked mask stops", (not clear) and t == 0.0)

eng = make_engine()
mask = np.zeros((S, S), np.uint8); mask[:, 120:140] = 1   # 20px < 0.28*256
a, t, clear, _ = eng.steer_from_mask(mask)
check("corridor narrower than cart is blocked", not clear)

# fork: the junction command decides which branch we follow
mask = np.zeros((S, S), np.uint8); mask[:, 10:100] = 1; mask[:, 156:246] = 1
eng = make_engine(); aL, _, cL, _ = eng.steer_from_mask(mask, "LEFT")
eng = make_engine(); aR, _, cR, _ = eng.steer_from_mask(mask, "RIGHT")
check("fork: LEFT and RIGHT diverge", cL and cR and aL < -0.2 and aR > 0.2,
      f"L={aL:.2f} R={aR:.2f}")

# OPEN junction: continuous tarmac, so every band sees ONE wide run and
# _pick_run has nothing to choose between. This is the campus-plaza case that
# used to give identical steering for all three commands and drive the cart
# straight through every turn.
wide = np.zeros((S, S), np.uint8); wide[:, 8:248] = 1
eng = make_engine(); aL, _, cL, _ = eng.steer_from_mask(wide, "LEFT")
eng = make_engine(); aR, _, cR, _ = eng.steer_from_mask(wide, "RIGHT")
eng = make_engine(); aS, _, cS, _ = eng.steer_from_mask(wide, "STRAIGHT")
check("open junction: LEFT steers left", cL and aL < -0.2, f"a={aL:+.2f}")
check("open junction: RIGHT steers right", cR and aR > 0.2, f"a={aR:+.2f}")
check("open junction: STRAIGHT stays centred", cS and abs(aS) < 0.05,
      f"a={aS:+.2f}")
check("open junction: the three commands differ",
      abs(aL - aR) > 0.4 and abs(aL - aS) > 0.2 and abs(aR - aS) > 0.2,
      f"L={aL:+.2f} S={aS:+.2f} R={aR:+.2f}")

# The bias must never aim the cart closer to a boundary than it can fit. That
# invariant lives in _target_x and is checked here directly: at mask level the
# width requirement shrinks with distance (perspective), so a fixed-pixel
# corridor is genuinely wide in the far bands and SHOULD bias there.
check("bias keeps full clearance from the commanded edge",
      _target_x((0, 200), "LEFT", 40, 1.0) == 20 and
      _target_x((0, 200), "RIGHT", 40, 1.0) == 180)
check("bias is proportional between centre and edge",
      _target_x((0, 200), "LEFT", 40, 0.5) == 60)
check("corridor only as wide as the cart cannot bias",
      _target_x((0, 40), "LEFT", 40, 1.0) == 20 and
      _target_x((0, 30), "LEFT", 40, 1.0) == 15)
check("no command aims at the centre",
      _target_x((0, 200), None, 40, 1.0) == 100 and
      _target_x((0, 200), "STRAIGHT", 40, 1.0) == 100)

# and a turn command must not override an actual obstacle
eng = make_engine()
check("turn command cannot invent a corridor",
      not eng.steer_from_mask(np.zeros((S, S), np.uint8), "LEFT")[2])

eng = make_engine(junction_bias=0.0)
a0, _, _, _ = eng.steer_from_mask(wide, "LEFT")
check("junction_bias=0 restores centring", abs(a0) < 0.05, f"a={a0:+.2f}")

# --- the camera decides WHEN to turn, GPS only decides which way ---
# A corridor that runs straight ahead with no side opening: a LEFT command
# arrives (GPS fired early, several metres of error) but there is nowhere to
# turn yet, so the cart must keep centring rather than aim at the verge.
corridor = np.zeros((S, S), np.uint8); corridor[:, 88:168] = 1
eng = make_engine(); a_early, _, c_early, d_early = eng.steer_from_mask(corridor, "LEFT")
eng = make_engine(); a_plain, _, _, _ = eng.steer_from_mask(corridor)
check("no arm in sight: turn command is not obeyed yet",
      c_early and not d_early["armed"] and abs(a_early - a_plain) < 0.05,
      f"cmd={a_early:+.2f} plain={a_plain:+.2f}")

# the same corridor, now with a genuine left opening near the cart
with_arm = corridor.copy()
with_arm[S - 60:, 4:168] = 1
eng = make_engine(); a_arm, _, _, d_arm = eng.steer_from_mask(with_arm, "LEFT")
check("left arm visible: the command is obeyed",
      d_arm["arms"]["left"] and d_arm["armed"] and a_arm < a_plain - 0.1,
      f"cmd={a_arm:+.2f} plain={a_plain:+.2f}")
check("a left arm is not mistaken for a right one",
      not d_arm["arms"]["right"])

# open plaza: no arm stands out, but there is room to move — obey the command
eng = make_engine(); a_open, _, _, d_open = eng.steer_from_mask(wide, "LEFT")
check("open ground obeys the command without a distinct arm",
      d_open["arms"]["open_ground"] and d_open["armed"] and a_open < -0.2,
      f"a={a_open:+.2f}")

# --- slew limit: a single bad frame must not snap the wheels to full lock ---
# a centred corridor to settle on, then a hard-right one the geometry wants a
# large positive angle for
centred = np.zeros((S, S), np.uint8); centred[:, 88:168] = 1
hard = np.zeros((S, S), np.uint8); hard[:, 170:250] = 1
eng = make_engine(max_steer_rate=0.0)
eng.steer_from_mask(centred)                      # establish a previous angle
a_unlimited, _, _, _ = eng.steer_from_mask(hard)

eng = make_engine(max_steer_rate=1.2)
eng.steer_from_mask(centred)
time.sleep(0.05)                                   # a realistic short dt
a_limited, _, _, _ = eng.steer_from_mask(hard)
check("slew limit caps a one-frame jump", abs(a_limited) < abs(a_unlimited),
      f"limited={a_limited:+.2f} unlimited={a_unlimited:+.2f}")

# ...but a SUSTAINED turn must still reach the angle it wants
eng = make_engine(max_steer_rate=1.2)
eng.steer_from_mask(centred)
for _ in range(12):
    time.sleep(0.05)
    a_sustained, _, _, _ = eng.steer_from_mask(hard)
check("slew limit does not block a sustained turn",
      abs(a_sustained - a_unlimited) < 0.05,
      f"sustained={a_sustained:+.2f} target={a_unlimited:+.2f}")

# a blocked corridor must report instantly, not ramp down through the limit
eng = make_engine(max_steer_rate=1.2)
eng.steer_from_mask(hard)
time.sleep(0.05)
a_block, t_block, c_block, _ = eng.steer_from_mask(np.zeros((S, S), np.uint8))
check("blocked corridor is not rate limited",
      (not c_block) and a_block == 0.0 and t_block == 0.0)

eng = make_engine()
full = np.zeros((S, S), np.uint8); full[:, 88:168] = 1
_, t_full, _, _ = eng.steer_from_mask(full)
eng = make_engine()
short = np.zeros((S, S), np.uint8); short[200:, 88:168] = 1
_, t_short, _, _ = eng.steer_from_mask(short)
check("throttle scales with corridor depth", t_short < t_full,
      f"short={t_short:.2f} full={t_full:.2f}")

check("_runs finds contiguous segments",
      _runs(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == [(1, 3), (4, 7)])


# =====================================================================
section("ground-plane calibration is actually checked")
# =====================================================================
# Four correspondences fit a homography EXACTLY, so reprojection error is
# always ~0 and cannot detect a bad calibration. It used to be reported as
# accuracy, which called a broken calibration perfect. These are the checks
# that replaced it — the planner compares gaps against a 28 cm cart, so a
# silently wrong homography makes its clearance maths meaningless.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from calibrate_ground_plane import sanity_problems          # noqa: E402

_G = [(0.5, 0.75), (0.5, -0.75), (2.5, -0.75), (2.5, 0.75)]
_OK = [(150.0, 460.0), (490.0, 460.0), (390.0, 300.0), (250.0, 300.0)]

check("correct clicks raise no complaint", not sanity_problems(_OK, _G))
check("left/right swap is caught",
      bool(sanity_problems([_OK[1], _OK[0], _OK[2], _OK[3]], _G)))
check("near/far swap is caught",
      bool(sanity_problems([_OK[3], _OK[2], _OK[1], _OK[0]], _G)))
check("a crossing click order is caught",
      bool(sanity_problems([_OK[0], _OK[2], _OK[1], _OK[3]], _G)))

# A plain misclick keeps the ordering valid, so geometry cannot see it — only
# an independent fifth marker can. This asserts that gap is real, so nobody
# later assumes the geometry check is sufficient on its own.
_shift = list(_OK); _shift[2] = (360.0, 310.0)
check("a 30px misclick is NOT caught by geometry alone",
      not sanity_problems(_shift, _G))

_H_true = cv2.findHomography(np.array(_OK, np.float32),
                             np.array(_G, np.float32))[0]
_v_img = cv2.perspectiveTransform(np.array([[[1.5, 0.0]]], np.float32),
                                  np.linalg.inv(_H_true))[0][0]


def _verify_cm(clicks):
    H = cv2.findHomography(np.array(clicks, np.float32),
                           np.array(_G, np.float32))[0]
    got = cv2.perspectiveTransform(np.array([[_v_img]], np.float32), H)[0][0]
    return 100.0 * float(np.hypot(got[0] - 1.5, got[1] - 0.0))


check("fifth point reads ~0 on a good calibration", _verify_cm(_OK) < 1.0,
      f"{_verify_cm(_OK):.2f} cm")
check("fifth point catches the misclick geometry missed",
      _verify_cm(_shift) > 5.0, f"{_verify_cm(_shift):.2f} cm")


# =====================================================================
section("textured surfaces (grass pavers, dappled shade)")
# =====================================================================
# Half the area of a turf block is grass. Without mask closing the corridor
# shatters into strips too narrow to pass the width test, and the cart decides
# there is no path — on a surface that is perfectly drivable.
speckled = grass_paver_mask(S, 88, 168)

eng = make_engine(mask_close_px=0)          # closing disabled
raw = eng.clean_mask(speckled)
_, _, clear_raw, _ = eng.steer_from_mask(raw)
check("grass paver WITHOUT closing is (wrongly) blocked", not clear_raw)

eng = make_engine(mask_close_px=9)          # closing enabled
closed = eng.clean_mask(speckled)
a, t, clear, _ = eng.steer_from_mask(closed)
check("grass paver WITH closing is drivable", clear and t > 0,
      f"a={a:+.2f} t={t:.2f}")
check("closing keeps the cart centred on the path", abs(a) < 0.2, f"a={a:+.2f}")

# closing must NOT invent a path where there is none, or bridge across a
# genuine obstacle sitting in the middle of the corridor
eng = make_engine(mask_close_px=9)
check("closing does not invent a corridor",
      not eng.steer_from_mask(eng.clean_mask(np.zeros((S, S), np.uint8)))[2])

wide_gap = np.zeros((S, S), np.uint8)
wide_gap[:, 20:110] = 1          # left branch
wide_gap[:, 146:236] = 1         # right branch, 36 px obstacle between
closed_gap = eng.clean_mask(wide_gap)
check("closing does not bridge a real obstacle",
      closed_gap[S // 2, 128] == 0)


# =====================================================================
section("SegPilot fails closed")
# =====================================================================
pilot = SegPilot.__new__(SegPilot)
pilot.max_result_age, pilot.max_frame_age = 0.3, 0.3
pilot.image = pilot.nav_command = None
pilot.angle, pilot.throttle, pilot._corridor_clear = 0.7, 0.4, True
pilot.fps = 2.0
pilot.mask = None
pilot._last_image, pilot._frame_time = None, 0.0
pilot._result_time, pilot._last_stale_log = 0.0, 0.0

frame1 = np.zeros((120, 160, 3), np.uint8)
pilot._result_time = time.monotonic()
a, t, clear, _, _ = pilot.run_threaded(frame1)
check("fresh result drives", clear and t > 0, f"a={a:.2f} t={t:.2f}")

# a frozen camera part keeps returning the SAME array object
time.sleep(0.35)
pilot._result_time = time.monotonic()
a, t, clear, _, _ = pilot.run_threaded(frame1)
check("frozen camera stops the cart", (not clear) and t == 0.0 and a == 0.0)

frame2 = np.ones((120, 160, 3), np.uint8)
pilot._result_time = time.monotonic() - 5.0
a, t, clear, _, _ = pilot.run_threaded(frame2)
check("stalled inference stops the cart", (not clear) and t == 0.0)

pilot._result_time = time.monotonic()
a, t, clear, _, _ = pilot.run_threaded(np.full((120, 160, 3), 2, np.uint8))
check("recovers once healthy again", clear and t > 0)


# =====================================================================
section("YoloGuard fails closed")
# =====================================================================
g = YoloGuard.__new__(YoloGuard)
g.max_result_age, g.max_failures = 1.0, 3
g._stop = g._slow = False
g._boxes = []   # (track_id, x1, y1, x2, y2)
g._failures, g._result_time, g._last_stale_log = 0, time.monotonic(), 0.0
g.fps, g.image = 5.0, None

stop, slow, healthy, _, _ = g.run_threaded(frame1)
check("healthy guard reports clear", (not stop) and healthy)

g._failures = 3
stop, _, healthy, _, _ = g.run_threaded(frame1)
check("repeated inference errors force stop", stop and (not healthy))

g._failures, g._result_time = 0, time.monotonic() - 9.0
stop, _, healthy, _, _ = g.run_threaded(frame1)
check("stalled detector forces stop", stop and (not healthy))

# corridor geometry
g2 = YoloGuard.__new__(YoloGuard)
g2.corridor_bottom_frac, g2.horizon_frac = 0.75, 0.45
check("person dead ahead is in corridor", g2._in_corridor(0.5, 0.95))
check("person far to the side is not", not g2._in_corridor(0.05, 0.95))
check("detection above horizon ignored", not g2._in_corridor(0.5, 0.2))


# =====================================================================
section("UltrasonicArray fails closed")
# =====================================================================
arr = UltrasonicArray.__new__(UltrasonicArray)
NAMES = ["left", "center", "right"]
arr.history = {n: [] for n in NAMES}
arr.misses = {n: 0 for n in NAMES}
arr.median_n, arr.max_misses = 5, 4
arr.stop_cm, arr.caution_cm = 30.0, 80.0
arr.clear_count, arr._clear_streak = 5, 0
arr.dist = {n: None for n in NAMES}
arr.stop, arr.healthy, arr.bias = True, False, 0.0
arr.angles = {"left": 30.0, "center": 0.0, "right": -30.0}
arr.has_angles = True
arr._last_unhealthy_log = 0.0


def feed(readings, ticks=6):
    for _ in range(ticks):
        for n in NAMES:
            arr.dist[n] = arr._filtered(n, readings[n])
        arr._evaluate()


feed({n: NOTHING_IN_RANGE for n in NAMES})
check("alive sensors, clear path -> drives", (not arr.stop) and arr.healthy)

feed({"left": NOTHING_IN_RANGE, "center": 20.0, "right": NOTHING_IN_RANGE}, ticks=1)
check("object inside stop_cm -> stop", arr.stop and arr.healthy)

# The important one. An unplugged HC-SR04 never drives ECHO high; a working one
# with clear air ahead still does. Reading silence as "clear" would let the cart
# drive with no working proximity sensing at all.
feed({n: NO_RESPONSE for n in NAMES})
check("all sensors dead -> stop + unhealthy", arr.stop and (not arr.healthy))
check("dead sensors report None, not stale cm",
      all(arr.dist[n] is None for n in NAMES))

feed({n: NOTHING_IN_RANGE for n in NAMES})
check("recovers after reconnection", (not arr.stop) and arr.healthy)

feed({"left": NO_RESPONSE, "center": NOTHING_IN_RANGE, "right": NOTHING_IN_RANGE})
check("a single dead sensor is enough to stop", arr.stop and (not arr.healthy))

feed({n: 100.0 for n in NAMES})
arr.dist["center"] = arr._filtered("center", NO_RESPONSE)
check("one dropped ping is tolerated", arr.dist["center"] is not None)

feed({"left": 40.0, "center": 60.0, "right": NOTHING_IN_RANGE}, ticks=6)
check("caution bias steers to the open side", arr.bias > 0, f"bias={arr.bias:+.1f}")


# =====================================================================
section("BreakerDetect")
# =====================================================================
def striped(y0=190, h=240, w=320, band_h=26):
    img = np.full((h, w, 3), 110, np.uint8)
    for i, x in enumerate(range(0, w, 40)):
        c = (0, 220, 240) if i % 2 == 0 else (25, 25, 25)  # BGR yellow / black
        cv2.rectangle(img, (x, y0), (x + band_h + 14, y0 + band_h), c, -1)
    return img


det, band_y = detect_breaker(striped(y0=205))
check("striped breaker detected", det and band_y > 0.7, f"y={band_y:.2f}")
check("plain road: no detection",
      not detect_breaker(np.full((240, 320, 3), 110, np.uint8))[0])

# a yellow signboard or wall is not a breaker — stripes are the signal
blob = np.full((240, 320, 3), 110, np.uint8)
cv2.rectangle(blob, (60, 180), (260, 230), (0, 220, 240), -1)
check("solid yellow blob rejected", not detect_breaker(blob)[0])

near = cv2.cvtColor(striped(y0=205), cv2.COLOR_BGR2RGB)
far = cv2.cvtColor(striped(y0=130), cv2.COLOR_BGR2RGB)
check("breaker far ahead does not trigger creep", not BreakerDetect().run(far))

bd = BreakerDetect()
check("breaker at the bumper triggers creep", bd.run(near))
bd.run(np.full((240, 320, 3), 110, np.uint8))
check("creep latches while wheels cross", bd.run(None))

# Roadside grass is a near-perfect decoy: dry golden verge reads as breaker
# paint. On real rural footage this fired on 71% of frames until the search
# was confined to drivable ground. A breaker is painted ON THE ROAD.
verge = np.full((240, 320, 3), 110, np.uint8)
verge[:, :90] = (60, 170, 200)          # golden grass left
verge[:, 230:] = (60, 170, 200)         # and right
# the carriageway, with verges either side excluded
road_only = np.zeros((240, 320), np.uint8)
road_only[:, 20:300] = 1
check("grass verges are ignored once the drivable mask is supplied",
      not detect_breaker(verge, drivable_mask=road_only)[0])

# a real striped breaker spans the carriageway and must survive the filter
check("a real breaker on the road survives the mask filter",
      detect_breaker(striped(y0=205), drivable_mask=road_only)[0])

try:
    detect_breaker(np.zeros((4, 4, 3), np.uint8))
    detect_breaker(np.zeros((4, 4, 3), np.uint8), drivable_mask=np.ones((4, 4), np.uint8))
    check("degenerate input does not raise", True)
except Exception as exc:                                    # noqa: BLE001
    check("degenerate input does not raise", False, str(exc))


# =====================================================================
section("GpsNav fails closed")
# =====================================================================
nav = GpsNav.__new__(GpsNav)
nav._fix, nav.geofence, nav.fix_stale_secs = None, None, 3.0
nav.route, nav.route_index, nav.arrived = [], 0, False
nav._lock = threading.Lock()
nav.junction_radius_m, nav.turn_threshold_deg, nav.arrive_radius_m = 8.0, 30.0, 5.0
nav.graph, nav._pending_destination, nav._last_route_attempt = None, None, 0.0

check("no fix -> unsafe", not nav.run_threaded()[3])

nav._fix = (26.475, 73.115, time.monotonic() - 99.0)
check("stale fix -> unsafe", not nav.run_threaded()[3])

FENCE = [(26.470, 73.110), (26.470, 73.120), (26.480, 73.120), (26.480, 73.110)]
nav._fix, nav.geofence = (26.475, 73.115, time.monotonic()), FENCE
check("fresh fix inside fence -> safe", nav.run_threaded()[3])

nav._fix = (26.495, 73.115, time.monotonic())
check("outside fence -> unsafe", not nav.run_threaded()[3])

# a destination given before the receiver locks on must be held, not dropped
nav2 = GpsNav.__new__(GpsNav)
nav2._fix, nav2.geofence, nav2.fix_stale_secs = None, None, 4.0
nav2.route, nav2.route_index, nav2.arrived = [], 0, False
nav2._lock = threading.Lock()
nav2.junction_radius_m, nav2.turn_threshold_deg, nav2.arrive_radius_m = 12.0, 30.0, 8.0
nav2._last_route_attempt, nav2._pending_destination = 0.0, None
nav2.graph = object()            # graph loaded, but the receiver has no fix yet

check("destination with no fix is queued, not lost",
      nav2.set_destination(26.47, 73.11) is False
      and nav2._pending_destination == (26.47, 73.11))

nav2._resolve_pending()          # still no fix: must not crash, must stay queued
check("queued destination survives having no fix",
      nav2._pending_destination == (26.47, 73.11))

# A route arrives from the EC2 server already computed, so no graph or
# networkx is needed on the Pi at all.
# park the cart well away from every waypoint, so waypoint-advance logic does
# not consume the first one and muddy what these assertions are measuring
nav2._fix = (26.460, 73.100, time.monotonic())
nav2._pending_destination, nav2._route_key = None, None
nav2.route, nav2.route_index, nav2.arrived = [], 3, True

ROUTE_A = [[26.470, 73.110], [26.475, 73.115], [26.480, 73.120]]
nav2.run_threaded(route=ROUTE_A)
check("server route is adopted", len(nav2.route) == 3)
check("adopting a route resets progress",
      nav2.route_index == 0 and nav2.arrived is False)

nav2.route_index = 2                              # pretend we drove some of it
nav2.run_threaded(route=ROUTE_A)                  # same route republished
check("repeated route does not restart the run", nav2.route_index == 2)

ROUTE_B = [[26.470, 73.110], [26.490, 73.130]]
nav2.run_threaded(route=ROUTE_B)
check("a different route replaces the old one",
      len(nav2.route) == 2 and nav2.route_index == 0)

# This is the one that matters operationally: losing the server mid-delivery
# must not strand the cart. The route is already local.
nav2.route_index = 1
nav2.run_threaded(route=None)
check("server going away leaves the route running",
      len(nav2.route) == 2 and nav2.route_index == 1)

check("a degenerate route is refused", nav2.set_route([[26.47, 73.11]]) is False)

# with no graph at all, routing can never succeed, so queueing would be a lie
nav3 = GpsNav.__new__(GpsNav)
nav3.graph, nav3._pending_destination, nav3._fix = None, None, None
check("no graph -> refuses outright rather than queueing forever",
      nav3.set_destination(26.47, 73.11) is False
      and nav3._pending_destination is None)

check("point_in_polygon inside", point_in_polygon(26.475, 73.115, FENCE))
check("point_in_polygon outside", not point_in_polygon(26.490, 73.115, FENCE))
check("bearing due east is ~90", abs(bearing_deg(26.0, 73.0, 26.0, 73.01) - 90) < 1)
check("haversine ~111 km per degree",
      abs(haversine_m(26.0, 73.0, 27.0, 73.0) - 111195) < 500)


# =====================================================================
section("SafetyArbiter priority chain")
# =====================================================================
BASE = dict(seg_angle=0.5, seg_throttle=0.3, corridor_clear=True,
            sonar_stop=False, sonar_bias=0.0, sonar_healthy=True,
            yolo_stop=False, yolo_slow=False, yolo_healthy=True,
            breaker_active=False, nav_safe=True, nav_arrived=False,
            plan_angle=None, plan_throttle=None, plan_clear=False)

arb = SafetyArbiter(creep_throttle=0.14, mission_requires_gps=True,
                    require_sonar=True, require_yolo=True)
check("all healthy -> drives", arb.run(**BASE) == (0.5, 0.3))
check("sonar unhealthy -> stop", arb.run(**{**BASE, "sonar_healthy": False}) == (0.0, 0.0))
check("yolo unhealthy -> stop", arb.run(**{**BASE, "yolo_healthy": False}) == (0.0, 0.0))
check("sonar hard stop wins", arb.run(**{**BASE, "sonar_stop": True}) == (0.0, 0.0))
check("gps unsafe -> stop (fail closed)",
      arb.run(**{**BASE, "nav_safe": False}) == (0.0, 0.0))
check("person detected -> stop", arb.run(**{**BASE, "yolo_stop": True}) == (0.0, 0.0))
check("no drivable corridor -> stop",
      arb.run(**{**BASE, "corridor_clear": False}) == (0.0, 0.0))
check("arrived -> stop", arb.run(**{**BASE, "nav_arrived": True}) == (0.0, 0.0))
check("breaker mode -> straight and creep",
      arb.run(**{**BASE, "breaker_active": True}) == (0.0, 0.14))
check("yolo slow halves throttle",
      arb.run(**{**BASE, "yolo_slow": True}) == (0.5, 0.15))

angle, throttle = arb.run(**{**BASE, "sonar_bias": 0.5})
check("sonar bias applied and clipped to 1.0",
      angle == 1.0 and throttle == 0.3, f"angle={angle}")

# throttle floor for a sensorless brushless motor: lift commands it cannot act
# on, but never turn a stop into motion
arb_floor = SafetyArbiter(creep_throttle=0.10, mission_requires_gps=False,
                          require_sonar=False, require_yolo=False,
                          min_move_throttle=0.22)
check("floor lifts a too-low creep",
      arb_floor.run(**{**BASE, "breaker_active": True}) == (0.0, 0.22))
check("floor lifts a halved slow command",
      arb_floor.run(**{**BASE, "yolo_slow": True}) == (0.5, 0.22))
check("floor leaves an adequate command alone",
      arb_floor.run(**BASE) == (0.5, 0.3))
check("floor never turns a stop into motion",
      arb_floor.run(**{**BASE, "sonar_stop": True}) == (0.0, 0.0))
check("floor never overrides a blocked corridor",
      arb_floor.run(**{**BASE, "corridor_clear": False}) == (0.0, 0.0))

arb_indoor = SafetyArbiter(mission_requires_gps=False,
                           require_sonar=False, require_yolo=False)
check("indoor mode ignores GPS and absent layers",
      arb_indoor.run(**{**BASE, "nav_safe": False, "sonar_healthy": False,
                        "yolo_healthy": False}) == (0.5, 0.3))
check("sonar still stops in indoor mode",
      arb_indoor.run(**{**BASE, "sonar_stop": True}) == (0.0, 0.0))


# =====================================================================
section("LocalPlanner steers around obstacles")
# =====================================================================
# The point of the planner: a person on the path should be driven AROUND when
# there is room, not merely stopped at. Grids are built directly here, so no
# camera, homography or warp is involved.

def make_planner(**kw):
    p = LocalPlanner.__new__(LocalPlanner)
    p.cart_width_m = kw.get("cart_width_m", 0.28)
    p.wheelbase_m = 0.32
    p.safety_margin_m = kw.get("safety_margin_m", 0.12)
    p.horizon_m, p.res, p.lateral_m = 3.0, 0.05, 1.6
    p.rows, p.cols = int(3.0 / 0.05), int(2 * 1.6 / 0.05)
    p.n_candidates, p.max_steer_rad = 61, 0.52
    p.smoothness_weight, p.heading_weight = 0.35, 0.5
    p.throttle_cruise, p.throttle_creep = 0.30, 0.16
    p.min_clear_m = 0.45
    p._prev_steer = 0.0
    inflate = int(round((p.cart_width_m / 2 + p.safety_margin_m) / p.res))
    k = max(3, 2 * inflate + 1)
    p._inflate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    p.homography, p.enabled = None, True
    p.assumed_speed_ms = kw.get("assumed_speed_ms", 0.5)
    p.predict_horizon_s = 2.5
    p.track_window_s, p.min_track_speed_ms = 0.8, 0.25
    p._cart_radius = p.cart_width_m / 2 + p.safety_margin_m
    p._tracks = {}
    return p

planner = make_planner()
R, C = planner.rows, planner.cols

def open_path(width_m=2.0):
    g = np.zeros((R, C), np.uint8)
    half = int((width_m / 2) / planner.res)
    g[:, C // 2 - half:C // 2 + half] = 1
    return g

# clear path -> go straight, full clearance
g = open_path()
steer, clear_m, _ = planner.plan(g, goal_bias=0.0)
check("open path: drives straight", abs(steer) < 0.15, f"steer={steer:+.2f}")
check("open path: full clearance", clear_m > 2.5, f"clear={clear_m:.1f}m")

# person 1.5 m ahead, standing slightly RIGHT of centre, room on the left
def person_at(x_m, y_m, radius_m=0.25, base=None):
    g = open_path() if base is None else base.copy()
    row = int((planner.horizon_m - x_m) / planner.res)
    col = int((planner.lateral_m - y_m) / planner.res)
    cv2.circle(g, (col, row), int(radius_m / planner.res), 0, -1)
    return g

planner._prev_steer = 0.0
g = planner.inflate(person_at(1.5, -0.35))
steer, clear_m, _ = planner.plan(g, goal_bias=0.0)
# direction is what matters, not magnitude: if a small dodge already clears
# the way, a violent swerve would be the wrong answer
check("person on the right: steers LEFT around them",
      steer < 0 and clear_m > 1.5, f"steer={steer:+.2f} clear={clear_m:.1f}m")

# mirror it: person on the left -> go right
planner._prev_steer = 0.0
g = planner.inflate(person_at(1.5, 0.35))
steer, clear_m, _ = planner.plan(g, goal_bias=0.0)
check("person on the left: steers RIGHT around them",
      steer > 0 and clear_m > 1.5, f"steer={steer:+.2f} clear={clear_m:.1f}m")

# a person filling a NARROW path leaves no room -> must not squeeze past
planner._prev_steer = 0.0
narrow = person_at(1.2, 0.0, radius_m=0.35, base=open_path(width_m=1.0))
g = planner.inflate(narrow)
steer, throttle, clear, dist = planner.run.__wrapped__(planner, None)     if hasattr(planner.run, "__wrapped__") else (None, None, None, None)
_, clear_m, _ = planner.plan(g, goal_bias=0.0)
check("blocked narrow path: no arc gets through",
      clear_m < 1.2, f"clear={clear_m:.1f}m")

# safety margin must actually bite: a gap barely wider than the cart is not
# passable once the margin is added
planner_wide = make_planner(safety_margin_m=0.30)
gap = np.zeros((R, C), np.uint8)
gap_half = int((0.50 / 2) / planner_wide.res)      # 50 cm gap, cart is 28 cm
gap[:, C // 2 - gap_half:C // 2 + gap_half] = 1
_, clear_tight, _ = planner_wide.plan(planner_wide.inflate(gap), goal_bias=0.0)
check("margin refuses a gap only just wider than the cart",
      clear_tight < 1.0, f"clear={clear_tight:.1f}m")

# same gap, no margin demanded -> now it fits
planner_thin = make_planner(safety_margin_m=0.0)
_, clear_loose, _ = planner_thin.plan(planner_thin.inflate(gap), goal_bias=0.0)
check("without margin the same gap is passable",
      clear_loose > clear_tight, f"{clear_loose:.1f}m vs {clear_tight:.1f}m")

# goal bias should influence the choice when both ways are equally clear
planner._prev_steer = 0.0
sL, _, _ = planner.plan(open_path(width_m=3.0), goal_bias=-0.6)
planner._prev_steer = 0.0
sR, _, _ = planner.plan(open_path(width_m=3.0), goal_bias=0.6)
check("junction bias shifts the chosen arc", sL < sR, f"L={sL:+.2f} R={sR:+.2f}")


# =====================================================================
section("sonar fused into the planner grid")
# =====================================================================
# Four HC-SR04s are a crude range scan. They cannot see a scene the way a
# depth model can, but what they DO report is measured rather than inferred,
# and it costs no CPU — which on a 2 GB Pi is the deciding argument.

sp = make_planner()

# scan() must report only real returns. Silence is not evidence of clear
# ground: sound glances off angled surfaces and never comes back.
arr.dist = {"left": NOTHING_IN_RANGE, "center": 60.0, "right": None}
scan = arr.scan()
check("scan reports only measured returns", scan == [(0.0, 0.6)], f"{scan}")

arr.dist = {"left": 80.0, "center": 60.0, "right": 100.0}
scan = arr.scan()
check("scan carries bearing and metres",
      sorted(scan) == [(-30.0, 1.0), (0.0, 0.6), (30.0, 0.8)], f"{scan}")

# a return dead ahead must block the straight arc even on a mask that claims
# the path is completely clear — this is the case where vision is wrong and
# the sonar is right
clear_grid = open_path(width_m=2.0)
sp._prev_steer = 0.0
_, clear_before, _ = sp.plan(sp.inflate(clear_grid), goal_bias=0.0)
blocked = sp.inflate(clear_grid, sp.sonar_to_grid([(0.0, 0.8)]))
sp._prev_steer = 0.0
steer_after, clear_after, _ = sp.plan(blocked, goal_bias=0.0)
check("sonar return blocks a path vision called clear",
      clear_after < clear_before, f"{clear_after:.1f}m vs {clear_before:.1f}m")

# one reading is an ARC, not a point: the sensor cannot say where in its beam
# the echo came from, so the whole cone must be treated as occupied
cone = sp.sonar_to_grid([(0.0, 1.0)], beam_deg=15.0)
row = int((sp.horizon_m - 1.0) / sp.res)
width = int(cone[row - 1:row + 2].sum(axis=0).astype(bool).sum())
check("a return marks the whole beam cone, not one cell", width >= 4,
      f"{width} cells wide")

# out-of-range and nonsense readings are dropped rather than drawn somewhere
check("readings past the horizon are ignored",
      sp.sonar_to_grid([(0.0, 99.0)]).sum() == 0)
check("non-positive readings are ignored",
      sp.sonar_to_grid([(0.0, 0.0)]).sum() == 0)

# an angled return blocks the side it came from, not the centre
side = sp.sonar_to_grid([(30.0, 1.0)])
left_half = side[:, :sp.cols // 2].sum()
right_half = side[:, sp.cols // 2:].sum()
check("a left-bearing return marks the left side",
      left_half > right_half, f"L={left_half} R={right_half}")


# =====================================================================
section("unobserved ground is not blocked ground")
# =====================================================================
# A forward camera cannot see the ground at its own bumper — the nearest
# visible point is metres out. Reading those empty cells as obstacles made
# every arc collide at range zero: the cart sat still reporting "blocked"
# when it actually meant "no data here". Caught on real dashcam footage.

bp = make_planner()
# a realistic forward-camera homography: the image maps to ground BEYOND the
# bumper, exactly like a real mount — the near strip is never imaged
bp.homography = cv2.getPerspectiveTransform(
    np.float32([[20, 250], [236, 250], [150, 140], [106, 140]]),
    np.float32([[0.6, 1.2], [0.6, -1.2], [2.8, -1.2], [2.8, 1.2]]))
tiny = np.ones((256, 256), np.uint8)          # everything the camera sees is drivable
g = bp.build_grid(tiny)
check("grid is not all-blocked when the view is all-drivable", g.mean() > 0.5,
      f"{100*g.mean():.0f}% free")
check("build_grid records which cells were actually imaged",
      hasattr(bp, "observed") and bp.observed.shape == g.shape)
check("cells outside the camera's view are free, not blocked",
      g[bp.observed == 0].all() if (bp.observed == 0).any() else True)

# and a genuinely blocked observed cell must still read blocked
half = np.ones((256, 256), np.uint8); half[:, :128] = 0
g2 = bp.build_grid(half)
blocked_and_seen = ((g2 == 0) & (bp.observed == 1)).sum()
check("observed non-drivable ground still blocks", blocked_and_seen > 0,
      f"{blocked_and_seen} cells")


# =====================================================================
section("tracking and predictive planning")
# =====================================================================
# Blocking everywhere a moving obstacle MIGHT go would make anyone walking
# parallel to the cart close the whole path. The question asked instead is
# "will we and this person be in the same place at the same moment".

pp = make_planner(assumed_speed_ms=0.5)
open2 = pp.inflate(open_path(width_m=3.0))

# nothing moving: full clearance
pp._prev_steer = 0.0
_, base_clear, _ = pp.plan(open2, goal_bias=0.0, moving=None)
check("no moving obstacles: arc is clear", base_clear > 2.5, f"{base_clear:.1f}m")

# someone walking ACROSS our path, arriving where we will be when we get there
# meets us at t=2.0 s, x=1.0 m — inside the prediction horizon. Further out
# than that the arc is planned against static obstacles only, because a
# pedestrian's velocity six seconds from now is not information.
crosser = [{"x": 1.0, "y": -1.0, "vx": 0.0, "vy": 0.5, "radius": 0.3, "id": 1}]
pp._prev_steer = 0.0
_, cross_clear, _ = pp.plan(open2, goal_bias=0.0, moving=crosser)
check("a crossing pedestrian shortens the straight arc",
      cross_clear < base_clear, f"{cross_clear:.1f}m vs {base_clear:.1f}m")

# someone walking AWAY along our path must NOT block us — this is the case a
# naive swept-volume approach gets wrong
leaver = [{"x": 1.0, "y": 0.0, "vx": 2.0, "vy": 0.0, "radius": 0.3, "id": 2}]
pp._prev_steer = 0.0
_, leave_clear, _ = pp.plan(open2, goal_bias=0.0, moving=leaver)
check("someone walking away does not block the path",
      leave_clear > cross_clear, f"{leave_clear:.1f}m vs {cross_clear:.1f}m")

# a stationary object in the same spot blocks regardless of timing
stander = [{"x": 1.2, "y": 0.0, "vx": 0.0, "vy": 0.0, "radius": 0.3, "id": 3}]
pp._prev_steer = 0.0
_, stand_clear, _ = pp.plan(open2, goal_bias=0.0, moving=stander)
check("a stationary obstacle still blocks", stand_clear < base_clear,
      f"{stand_clear:.1f}m")

# --- velocity estimation happens on the GROUND, not in the image ---
tp = make_planner()
tp.homography = np.eye(3, dtype=np.float32)   # identity: pixels == metres here
shape = (240, 320, 3)

# same track id seen twice, 0.5 s apart, moved 0.5 m in x -> 1.0 m/s
t0 = 1000.0
tp.update_tracks([(7, 100, 100, 140, 150)], shape, t0)
moving = tp.update_tracks([(7, 100, 100, 140, 200)], shape, t0 + 0.5)
check("velocity is estimated from repeat sightings", len(moving) == 1,
      f"{len(moving)} moving")
if moving:
    # with an identity homography one "metre" is one pixel, so the foot point
    # moving 50 px over 0.5 s reads as 100 units/s
    speed = (moving[0]["vx"] ** 2 + moving[0]["vy"] ** 2) ** 0.5
    check("velocity is measured on the ground plane, not in the image",
          abs(speed - 100.0) < 1.0, f"speed={speed:.1f}")

# an object that barely moves is left to the occupancy grid
tp2 = make_planner()
tp2.homography = np.eye(3, dtype=np.float32)
tp2.update_tracks([(9, 100, 100, 140, 150)], shape, t0)
still = tp2.update_tracks([(9, 100, 100, 140, 150.01)], shape, t0 + 0.5)
check("a near-stationary track is not treated as moving", still == [],
      f"{still}")

# a box with no track id yet has no usable velocity and must be skipped
tp3 = make_planner()
tp3.homography = np.eye(3, dtype=np.float32)
tp3.update_tracks([(-1, 100, 100, 140, 150)], shape, t0)
check("untracked boxes yield no velocity",
      tp3.update_tracks([(-1, 100, 100, 140, 200)], shape, t0 + 0.5) == [])

# tracks that leave the frame are forgotten rather than accumulating forever
tp4 = make_planner()
tp4.homography = np.eye(3, dtype=np.float32)
tp4.update_tracks([(1, 10, 10, 20, 20), (2, 30, 30, 40, 40)], shape, t0)
tp4.update_tracks([(1, 10, 10, 20, 20)], shape, t0 + 0.1)
check("vanished tracks are dropped", list(tp4._tracks) == [1], f"{list(tp4._tracks)}")


# =====================================================================
section("bottom crop + calibration agreement")
# =====================================================================
# The cart's own bodywork in frame is a PERCEPTION failure, not wasted pixels:
# the model calls it `vehicle-car` and that label bleeds upward over the road.
# Measured on a dashcam frame with the bonnet in the bottom 16% — the tarmac
# ahead read `vehicle-car` 72%, drivable 8%, and the planner steered -0.65 off
# the road; cropped, 52% and -0.05. These tests pin the mechanics of the knob.

ce = make_engine()
ce.crop_bottom = 0.0
full = np.zeros((360, 640, 3), np.uint8)
check("crop_bottom 0 is a no-op", ce.crop(full).shape == full.shape)

ce.crop_bottom = 0.16
check("crop_bottom removes the bottom strip",
      ce.crop(full).shape[0] == 302, f"{ce.crop(full).shape[0]} rows")
check("crop_bottom keeps full width", ce.crop(full).shape[1] == 640)
check("crop_bottom keeps the TOP rows, not the bottom ones",
      np.array_equal(ce.crop(np.arange(360, dtype=np.uint8)
                             .reshape(360, 1, 1) * np.ones((1, 4, 3), np.uint8))[0],
                     np.zeros((4, 3), np.uint8)))

ce.crop_bottom = 0.5
check("a large crop still leaves a frame", ce.crop(full).shape[0] == 180)

rejected = 0
for v in (-0.1, 0.9, 1.0, 2.0):
    try:
        SegEngine.__init__(SegEngine.__new__(SegEngine), "x", "y", crop_bottom=v)
    except ValueError:
        rejected += 1
    except Exception:
        pass          # any earlier failure (model load) means validation order
check("crop_bottom outside [0, 0.9) is rejected", rejected == 4, f"{rejected}/4")

# A crop changes which pixel means which patch of ground, so a homography
# measured without it is wrong with it — and wrong SILENTLY, producing a
# plausible grid with every distance skewed. Refusing is the safe outcome:
# no homography means the cart falls back to seg_pilot, which needs no metres.
import json as _json
import tempfile as _tempfile


def _cal(crop):
    p = Path(_tempfile.gettempdir()) / f"_test_homography_{crop}.json"
    body = {"homography": np.eye(3).tolist()}
    if crop is not None:
        body["crop_bottom"] = crop
    p.write_text(_json.dumps(body))
    return str(p)


check("matching crop loads the homography",
      LocalPlanner._load_homography(_cal(0.16), 0.16) is not None)
check("mismatched crop REFUSES the homography",
      LocalPlanner._load_homography(_cal(0.0), 0.16) is None)
check("mismatch the other way is refused too",
      LocalPlanner._load_homography(_cal(0.16), 0.0) is None)
check("a calibration with no crop field means no crop",
      LocalPlanner._load_homography(_cal(None), 0.0) is not None)
check("old calibration + new crop is refused, not assumed",
      LocalPlanner._load_homography(_cal(None), 0.16) is None)

check("a crop mismatch DISABLES the planner rather than skewing it",
      not LocalPlanner(_cal(0.0), crop_bottom=0.16).enabled)
check("agreement leaves the planner enabled",
      LocalPlanner(_cal(0.16), crop_bottom=0.16).enabled)


# =====================================================================
section("SegPilot publishes the mask the planner needs")
# =====================================================================
# LocalPlanner takes seg/mask as its ONLY perception input. If SegPilot never
# assigns it, the planner receives None every tick and silently passes
# seg_pilot's steering straight back out — obstacle avoidance reads as enabled
# in the config and never once runs. This is invisible on a bench, so pin it.

mp = SegPilot.__new__(SegPilot)
mp.max_result_age, mp.max_frame_age = 5.0, 5.0
mp.angle, mp.throttle, mp._corridor_clear = 0.1, 0.2, True
mp.fps = 2.0
mp._last_image, mp._frame_time = None, 0.0
mp._result_time, mp._last_stale_log = time.monotonic(), 0.0
mp.image = mp.nav_command = None
mp.mask = np.ones((16, 16), np.uint8)
_, _, _, _, published = mp.run_threaded(np.zeros((120, 160, 3), np.uint8))
check("a healthy tick publishes the mask", published is not None)

mp._result_time = time.monotonic() - 99.0
_, _, _, _, stale = mp.run_threaded(np.ones((120, 160, 3), np.uint8))
check("a stale tick publishes None, not a stale mask", stale is None)

_src = (Path(__file__).resolve().parent.parent / "mycar" / "parts"
        / "seg_pilot.py").read_text(encoding="utf-8")
check("update() actually assigns self.mask", "self.mask = mask" in _src)


# =====================================================================
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL PASS")
