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

from parts.seg_pilot import SegEngine, SegPilot, _runs                # noqa: E402
from parts.yolo_guard import YoloGuard                                # noqa: E402
from parts.ultrasonic import UltrasonicArray, NOTHING_IN_RANGE, NO_RESPONSE  # noqa: E402
from parts.breaker_detect import BreakerDetect, detect_breaker        # noqa: E402
from parts.gps_nav import (GpsNav, point_in_polygon, bearing_deg,     # noqa: E402
                           haversine_m)
from parts.safety_arbiter import SafetyArbiter                        # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAILURES.append(name)


def section(title):
    print(f"\n--- {title} ---")


def make_engine(mask_close_px=9):
    """A SegEngine with the steering parameters but no ONNX session."""
    e = SegEngine.__new__(SegEngine)
    e.kp, e.kd = 1.2, 0.0
    e.bands, e.roi_top = 5, 0.4
    e.corridor_frac_bottom = 0.28
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
pilot._last_image, pilot._frame_time = None, 0.0
pilot._result_time, pilot._last_stale_log = 0.0, 0.0

frame1 = np.zeros((120, 160, 3), np.uint8)
pilot._result_time = time.monotonic()
a, t, clear, _ = pilot.run_threaded(frame1)
check("fresh result drives", clear and t > 0, f"a={a:.2f} t={t:.2f}")

# a frozen camera part keeps returning the SAME array object
time.sleep(0.35)
pilot._result_time = time.monotonic()
a, t, clear, _ = pilot.run_threaded(frame1)
check("frozen camera stops the cart", (not clear) and t == 0.0 and a == 0.0)

frame2 = np.ones((120, 160, 3), np.uint8)
pilot._result_time = time.monotonic() - 5.0
a, t, clear, _ = pilot.run_threaded(frame2)
check("stalled inference stops the cart", (not clear) and t == 0.0)

pilot._result_time = time.monotonic()
a, t, clear, _ = pilot.run_threaded(np.full((120, 160, 3), 2, np.uint8))
check("recovers once healthy again", clear and t > 0)


# =====================================================================
section("YoloGuard fails closed")
# =====================================================================
g = YoloGuard.__new__(YoloGuard)
g.max_result_age, g.max_failures = 1.0, 3
g._stop = g._slow = False
g._failures, g._result_time, g._last_stale_log = 0, time.monotonic(), 0.0
g.fps, g.image = 5.0, None

stop, slow, healthy, _ = g.run_threaded(frame1)
check("healthy guard reports clear", (not stop) and healthy)

g._failures = 3
stop, _, healthy, _ = g.run_threaded(frame1)
check("repeated inference errors force stop", stop and (not healthy))

g._failures, g._result_time = 0, time.monotonic() - 9.0
stop, _, healthy, _ = g.run_threaded(frame1)
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

try:
    detect_breaker(np.zeros((4, 4, 3), np.uint8))
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
            breaker_active=False, nav_safe=True, nav_arrived=False)

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
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL PASS")
