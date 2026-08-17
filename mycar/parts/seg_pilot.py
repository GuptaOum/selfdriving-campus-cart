"""
SegPilot — drivable-area segmentation pilot (the openpilot-style driver).

Runs a pretrained SegFormer-B0 (sidewalk-semantic) exported to INT8 ONNX,
collapses the classes to a binary drivable mask, and steers geometrically:
band centroids -> centerline -> proportional steering. Zero training data.

The pure-inference/steering logic lives in SegEngine so the offline bench
(scripts/vision_bench.py) exercises exactly the code the car runs.
"""
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization — what SegFormer was trained with
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegEngine:
    """ONNX segmentation + geometric steering. No DonkeyCar dependencies."""

    def __init__(self, onnx_path, labels_path,
                 kp=1.2, kd=0.3,
                 bands=5, roi_top=0.4,
                 corridor_frac_bottom=0.28,
                 throttle_cruise=0.30, throttle_creep=0.16,
                 mask_close_px=9, junction_bias=0.7, max_steer_rate=1.2):
        """
        :param onnx_path: segformer_sidewalk_int8.onnx
        :param labels_path: segformer_labels.json (from export_models.py)
        :param kp/kd: steering gains on normalized lateral offset
        :param bands: horizontal bands in the ROI for centroid extraction
        :param roi_top: fraction of image height where the ROI starts
        :param corridor_frac_bottom: min drivable run width at the BOTTOM band,
               as a fraction of image width (bot width + margin after homography
               calibration; 0.28 is a conservative pre-calibration default).
               Requirement shrinks linearly toward the top band (perspective).
        :param mask_close_px: morphological closing kernel, in mask pixels.
               Bridges gaps in a speckled mask so the corridor reads as one
               region. Needed for grass-paver / turf block (concrete grid with
               grass growing through), gravel with weeds, and dappled shade
               under trees — all of which segment as a checkerboard that would
               otherwise be rejected as dozens of too-narrow runs. Set to 0 to
               disable; raise it if your surface has bigger gaps.
        :param junction_bias: how hard a LEFT/RIGHT command pulls the aim point
               toward that side of a wide corridor, 0 (ignore the command) to
               1 (hug the edge at minimum clearance). Only matters where the
               corridor is wider than the cart needs; see _target_x.
        :param max_steer_rate: ceiling on how fast steering may change, in
               steering units per SECOND. Per second and not per frame because
               inference rate varies with load, and a per-frame limit would
               silently tighten whenever the Pi got busy. 0 disables.
        """
        import onnxruntime as ort  # deferred: not needed for unit tests of steering

        meta = json.loads(Path(labels_path).read_text())
        self.drivable_ids = np.array(meta["drivable_ids"], dtype=np.int64)
        self.input_size = int(meta.get("input_size", 256))
        if self.drivable_ids.size == 0:
            # would yield an all-zero mask, indistinguishable from "the model
            # sees no road anywhere" — catch the config error, not the symptom
            raise ValueError(
                f"{labels_path} lists no drivable class ids. Re-run "
                "scripts/export_models.py; check DRIVABLE_CLASS_NAMES matches "
                "the checkpoint's label strings (they carry a 'flat-' prefix).")
        logger.info("drivable classes: %s",
                    [meta["id2label"][str(i)] for i in meta["drivable_ids"]]
                    if "id2label" in meta else list(meta["drivable_ids"]))

        so = ort.SessionOptions()
        so.intra_op_num_threads = 3  # leave a core for the rest of the loop
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.kp, self.kd = kp, kd
        self.bands, self.roi_top = bands, roi_top
        self.corridor_frac_bottom = corridor_frac_bottom
        self.throttle_cruise, self.throttle_creep = throttle_cruise, throttle_creep
        self.mask_close_px = mask_close_px
        self.junction_bias = junction_bias
        self.max_steer_rate = max_steer_rate
        self._prev_angle = 0.0
        self._close_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (mask_close_px, mask_close_px))
            if mask_close_px and mask_close_px > 1 else None)
        self._prev_offset = 0.0
        self._prev_time = None
        self._prev_centroid_frac = 0.5

    def clean_mask(self, mask):
        """
        Bridge small non-drivable gaps so a textured surface reads as one
        corridor rather than a field of narrow strips.

        Grass-paver blocks are the motivating case: roughly half the area is
        grass growing through a concrete grid, so the raw mask is a
        checkerboard. Every strip then fails the corridor-width test and the
        cart decides there is no path at all. Closing (dilate then erode)
        fills holes smaller than the kernel while leaving the outer edges of
        the path where they are — so a genuine obstacle or a real path
        boundary still reads correctly.
        """
        if self._close_kernel is None:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

    def infer_mask(self, frame_bgr):
        """frame (any size, BGR) -> binary drivable mask at input_size x input_size."""
        s = self.input_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = img.transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input_name: tensor})[0]
        # SegFormer logits come out at 1/4 resolution; argmax there, upsample nearest
        classes = np.argmax(logits[0], axis=0).astype(np.int64)
        drivable = np.isin(classes, self.drivable_ids).astype(np.uint8)
        drivable = cv2.resize(drivable, (s, s), interpolation=cv2.INTER_NEAREST)
        return self.clean_mask(drivable)

    def steer_from_mask(self, mask, nav_command=None):
        """
        Binary mask -> (angle, throttle, corridor_clear, debug dict).
        nav_command: None/'STRAIGHT'/'LEFT'/'RIGHT' — junction bias from GPS.
        angle in [-1, 1] (DonkeyCar convention), throttle in [0, 1].
        """
        h, w = mask.shape
        roi_y0 = int(h * self.roi_top)
        band_h = (h - roi_y0) // self.bands

        # Pass 1: find the corridor in each band. The nav command is applied in
        # pass 2, because whether an arm exists can only be judged once the
        # whole set of bands is known — and that decides whether to bias at all.
        band_runs = []          # [(band_idx, (lo, hi))] nearest first
        required_px = {}
        for b in range(self.bands):
            # band 0 = bottom (nearest), band N-1 = top of ROI (farthest)
            y1 = h - b * band_h
            y0 = y1 - band_h
            band = mask[y0:y1, :]
            col_occ = band.mean(axis=0) > 0.5  # column drivable in this band

            runs = _runs(col_occ)
            if not runs:
                break  # blocked from this band outward
            # perspective: farther bands need proportionally narrower runs
            required = self.corridor_frac_bottom * w * (1.0 - 0.7 * b / max(self.bands - 1, 1))
            runs = [r for r in runs if (r[1] - r[0]) >= required]
            if not runs:
                break

            run = _pick_run(runs, nav_command, self._prev_centroid_frac * w)
            band_runs.append((b, run))
            required_px[b] = required

        # Pass 2: only bias toward a side the camera can actually see open.
        # If GPS says LEFT while the mask shows no left arm yet, the honest
        # answer is to keep driving straight and re-check next frame, not to
        # aim off the path on the strength of a several-metre position fix.
        arms = detect_arms(band_runs, required_px.get(0, 0.0), w)
        armed = (nav_command in ("LEFT", "RIGHT")
                 and (arms[nav_command.lower()] or arms["open_ground"]))
        effective_cmd = nav_command if armed else None

        centroids = []          # (band_idx, centroid_x_frac, run_width_px)
        deepest_clear = -1      # highest band index (nearest=0) whose corridor fits
        for b, run in band_runs:
            cx = _target_x(run, effective_cmd, required_px[b], self.junction_bias)
            centroids.append((b, cx / w, run[1] - run[0]))
            deepest_clear = b

        if not centroids:
            self._prev_offset = 0.0
            # not rate-limited: a blocked corridor must reach the arbiter at
            # once, and it is paired with zero throttle anyway
            self._prev_angle = 0.0
            return 0.0, 0.0, False, {"centroids": [], "deepest_clear": -1}

        # near bands dominate steering; far bands refine it
        weights = np.array([1.0 / (1 + c[0]) for c in centroids])
        weights /= weights.sum()
        centroid_frac = float(np.dot(weights, [c[1] for c in centroids]))
        self._prev_centroid_frac = centroid_frac

        offset = (centroid_frac - 0.5) * 2.0  # [-1, 1]
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time else None
        d_term = self.kd * (offset - self._prev_offset) / dt if dt and dt > 1e-3 else 0.0
        self._prev_offset, self._prev_time = offset, now

        angle = float(np.clip(self.kp * offset + d_term, -1.0, 1.0))

        # Slew limit. Measured on campus footage: every full-lock command was a
        # SINGLE isolated frame, always where the mask had thinned to one or
        # two surviving bands (9 of 599 frames, jumps up to 1.8 between
        # consecutive updates, each recovering immediately). One band means one
        # centroid decides the whole steering angle, and the D term then
        # amplifies the jump. On the cart that is a violent twitch of the
        # wheels on evidence that was gone by the next frame.
        #
        # Capping the RATE rather than the angle keeps genuine steering intact
        # — a junction turn of ~0.5 still completes well inside half a second —
        # while a one-frame outlier can only move the wheels a little before
        # the next frame corrects it. It deliberately does not smooth or
        # average: a sustained turn reaches full lock unimpeded, because every
        # frame keeps pushing the same way.
        if self.max_steer_rate and dt and dt > 1e-3:
            max_delta = self.max_steer_rate * dt
            angle = float(np.clip(angle, self._prev_angle - max_delta,
                                  self._prev_angle + max_delta))
        self._prev_angle = angle

        # throttle scales with how deep the corridor stays clear
        depth_frac = (deepest_clear + 1) / self.bands
        throttle = self.throttle_creep + (self.throttle_cruise - self.throttle_creep) * depth_frac

        debug = {"centroids": centroids, "deepest_clear": deepest_clear,
                 "offset": offset, "centroid_frac": centroid_frac,
                 "angle_raw": float(np.clip(self.kp * offset + d_term, -1, 1)),
                 "arms": arms, "nav_command": nav_command, "armed": armed}
        return angle, float(throttle), True, debug


def _runs(col_occ):
    """Contiguous True runs in a boolean column array -> [(start, end), ...]."""
    runs, start = [], None
    for i, v in enumerate(col_occ):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(col_occ)))
    return runs


def _pick_run(runs, nav_command, prev_cx):
    """Choose which drivable branch to follow — half of junction handling."""
    if nav_command == "LEFT":
        return runs[0]
    if nav_command == "RIGHT":
        return runs[-1]
    # STRAIGHT/None: stay on the branch closest to where we were heading
    return min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - prev_cx))


def detect_arms(band_runs, required_px, width):
    """
    Which side openings actually exist, judged from the mask alone.

    GPS decides WHICH way to turn; it is a poor judge of WHEN. It fires the
    command on distance to the waypoint, and with several metres of error that
    lands early or late — early means biasing toward an arm that is not in
    sight yet, which on a 5 m path aims the cart at the verge.

    The camera settles the timing. Take the farthest band that still has a
    corridor: that is the path continuing onward. Anything the NEAREST band
    reaches beyond that corridor's edges is ground that opens to the side —
    an arm. On a straight path the near band is only slightly wider than the
    far one (perspective), so nothing registers.

    :param band_runs: [(band_index, (lo, hi))] nearest first, as found by
           steer_from_mask after the width filter
    :param required_px: corridor width the cart needs at the bottom band
    :returns: {'left': bool, 'right': bool, 'left_px': float, 'right_px': float}
    """
    none = {"left": False, "right": False, "open_ground": False,
            "left_px": 0.0, "right_px": 0.0}
    if len(band_runs) < 2:
        return none  # nothing to compare against; one band proves nothing

    (_, (near_lo, near_hi)) = band_runs[0]
    (_, (far_lo, far_hi)) = band_runs[-1]

    left_px = far_lo - near_lo     # near band reaches further left than onward
    right_px = near_hi - far_hi

    # An arm has to be wide enough to drive into, or it is mask noise or the
    # ordinary perspective flare of a straight path.
    #
    # open_ground is the other way a turn can be safe. On a plaza the ground is
    # wide the whole way out, so nothing stands out as an "arm" — but there is
    # room to move over in any direction, which is exactly when a turn command
    # should be obeyed. Without this the gate would refuse to turn precisely
    # where turning is easiest.
    return {"left": left_px >= required_px,
            "right": right_px >= required_px,
            "open_ground": (far_hi - far_lo) >= 2.0 * required_px,
            "left_px": float(left_px), "right_px": float(right_px)}


def _target_x(run, nav_command, required, bias):
    """
    Where to aim WITHIN the chosen run — the other half of junction handling.

    _pick_run only helps when the junction arms are SEPARATE runs, i.e. when
    something non-drivable (grass, a kerb) sits between them. On an open paved
    junction the tarmac is continuous: every band reports one wide run, so
    LEFT, RIGHT and STRAIGHT all picked the same run, took its geometric
    middle, and produced identical steering — the cart sailed straight through
    every turn. That is what this fixes.

    A turn command means: stop centring, move to that side of the corridor and
    take the exit. So aim at a point offset from the commanded edge by the
    clearance the cart needs (`required`, which is cart width + margin at this
    band), then blend back toward the centre by `bias`.

    The clearance term is what keeps this safe. It is measured from the edge
    inward, so the aim point can never come closer to the boundary than the
    cart needs, however wide the junction is. And when the run is only just
    wide enough, the offset edge point IS the centre — the bias term vanishes
    on its own and a narrow path keeps centring exactly as before. No width
    threshold to tune.
    """
    lo, hi = run
    center = (lo + hi) / 2.0
    if nav_command not in ("LEFT", "RIGHT") or bias <= 0.0:
        return center
    if hi - lo <= required:
        return center  # no room to move over; hugging would clip the boundary
    half = required / 2.0
    edge = (lo + half) if nav_command == "LEFT" else (hi - half)
    return center + bias * (edge - center)


class SegPilot:
    """
    Threaded DonkeyCar part. Inference is slow (~0.5-1 s on a Pi 4), so it runs
    in its own thread at whatever rate it manages; run_threaded() returns the
    latest result to the 20 Hz vehicle loop instantly.

    Inputs:  cam/image_array (RGB, per DonkeyCar convention), nav/command
    Outputs: seg/angle, seg/throttle, seg/corridor_clear, seg/fps, seg/mask

    FAIL-CLOSED. The vehicle loop runs ~20x faster than inference, so it reuses
    each result many times — which is fine for one inference period and
    dangerous beyond it. Two watchdogs invalidate the result (corridor_clear
    goes False, which the arbiter turns into a stop):

      * result_age  — inference thread stalled, died, or slowed to a crawl
      * frame_age   — the camera stopped producing NEW frames. A frozen camera
                      part keeps handing back the same array, and without this
                      check the car would confidently drive on a photograph.
    """

    def __init__(self, onnx_path, labels_path,
                 max_result_age=1.5, max_frame_age=1.0, **engine_kwargs):
        """
        :param max_result_age: seconds a steering result stays valid. Must exceed
               your measured inference time or the car stops constantly — check
               the seg/fps output and set this to ~3x the period.
        :param max_frame_age: seconds without a new camera frame before stopping.
        """
        self.engine = SegEngine(onnx_path, labels_path, **engine_kwargs)
        self.max_result_age = max_result_age
        self.max_frame_age = max_frame_age

        self.image = None
        self.nav_command = None
        self.angle = 0.0
        self.throttle = 0.0
        self._corridor_clear = False
        self.fps = 0.0
        self.running = True

        self.mask = None
        self._last_image = None
        self._frame_time = 0.0
        self._result_time = 0.0
        self._last_stale_log = 0.0
        logger.info("SegPilot ready (input %dpx, result TTL %.1fs)",
                    self.engine.input_size, max_result_age)

    def update(self):
        while self.running:
            img = self.image
            if img is None:
                time.sleep(0.05)
                continue
            t0 = time.monotonic()
            try:
                # DonkeyCar images are RGB; engine expects BGR
                mask = self.engine.infer_mask(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.angle, self.throttle, self._corridor_clear, _ = \
                    self.engine.steer_from_mask(mask, self.nav_command)
                self._result_time = time.monotonic()
            except Exception:
                logger.exception("segmentation failed; treating as blocked")
                self.angle, self.throttle, self._corridor_clear = 0.0, 0.0, False
                self.mask = None
                time.sleep(0.2)  # back off rather than spin hot on a hard fault
            self.fps = 1.0 / max(time.monotonic() - t0, 1e-3)

    def _stale_reason(self, now):
        if now - self._result_time > self.max_result_age:
            return f"no inference for {now - self._result_time:.1f}s"
        if now - self._frame_time > self.max_frame_age:
            return f"no new camera frame for {now - self._frame_time:.1f}s"
        return None

    def run_threaded(self, image, nav_command=None):
        now = time.monotonic()
        # a frozen camera part returns the identical array object every tick
        if image is not None and image is not self._last_image:
            self._last_image = image
            self._frame_time = now
        self.image = image
        self.nav_command = nav_command

        reason = self._stale_reason(now)
        if reason:
            if now - self._last_stale_log > 2.0:
                logger.error("SegPilot output stale (%s) — reporting blocked", reason)
                self._last_stale_log = now
            # a stale mask must not reach the planner either
            return 0.0, 0.0, False, self.fps, None
        return (self.angle, self.throttle, self._corridor_clear,
                self.fps, self.mask)

    def shutdown(self):
        self.running = False
