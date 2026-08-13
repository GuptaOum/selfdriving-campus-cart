"""
YoloGuard — pretrained COCO person/obstacle detector as a safety layer.

YOLOv8n exported to NCNN (see scripts/export_models.py). It does NOT steer;
it only classifies "person (or large obstacle) in my corridor" into
slow / stop signals for the SafetyArbiter. HC-SR04 remains the last line —
this just reacts sooner and at range.

FAIL-CLOSED: a detector that is erroring or has gone stale reports "stop",
never "all clear". Silence from a pedestrian detector is not evidence of
no pedestrian.
"""
import logging
import time

logger = logging.getLogger(__name__)

# Classes that should stop a 1/8-scale cart when they are in its corridor.
# Matched by NAME, not index, so this survives a fine-tuned model whose class
# ordering differs from stock COCO. Names are the standard COCO80 spellings.
#
# 'cow' and 'dog' matter on an Indian campus specifically: strays wander onto
# paths, and both are in COCO already — no fine-tuning needed to catch them.
STOP_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "cow", "horse", "sheep",
}


class YoloGuard:
    """
    Threaded DonkeyCar part.

    Inputs:  cam/image_array (RGB)
    Outputs: yolo/stop, yolo/slow, yolo/healthy, yolo/fps, yolo/boxes

    Corridor test: bottom-center point of a detection must fall inside a
    trapezoid spanning [corridor_bottom_frac] of the width at the bottom of the
    frame, narrowing toward the horizon. bbox height fraction is the range
    proxy: tall box = close = stop, medium = slow.
    """

    def __init__(self, model_path, imgsz=320, conf=0.4,
                 corridor_bottom_frac=0.75, horizon_frac=0.45,
                 stop_height_frac=0.45, slow_height_frac=0.22,
                 max_result_age=2.5, max_failures=3):
        """
        :param max_result_age: seconds a detection result stays valid before the
               guard reports stop. Set above your measured inference period.
        :param max_failures: consecutive inference errors before declaring the
               guard unhealthy and forcing a stop.
        """
        from ultralytics import YOLO  # NCNN backend loads via ultralytics
        self.model = YOLO(str(model_path), task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.corridor_bottom_frac = corridor_bottom_frac
        self.horizon_frac = horizon_frac
        self.stop_height_frac = stop_height_frac
        self.slow_height_frac = slow_height_frac
        self.max_result_age = max_result_age
        self.max_failures = max_failures

        self.image = None
        self._stop = False
        self._slow = False
        self._boxes = []
        self.fps = 0.0
        self.running = True

        self._failures = 0
        self._result_time = 0.0
        self._last_stale_log = 0.0
        logger.info("YoloGuard ready (%s @ %dpx)", model_path, imgsz)

    def _in_corridor(self, cx_frac, y_frac):
        if y_frac < self.horizon_frac:
            return False  # above horizon = far away / not on ground
        # trapezoid: full corridor_bottom_frac wide at y=1.0, pinched at horizon
        t = (y_frac - self.horizon_frac) / (1.0 - self.horizon_frac)
        half_w = 0.5 * self.corridor_bottom_frac * (0.35 + 0.65 * t)
        return abs(cx_frac - 0.5) <= half_w

    def update(self):
        while self.running:
            img = self.image
            if img is None:
                time.sleep(0.05)
                continue
            t0 = time.monotonic()
            stop = slow = False
            found = []
            try:
                results = self.model.predict(img, imgsz=self.imgsz,
                                             conf=self.conf, verbose=False)
                h, w = img.shape[:2]
                names = results[0].names
                for box in results[0].boxes:
                    if names.get(int(box.cls)) not in STOP_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # every detection goes to the planner, in or out of the
                    # corridor — something beside the path still constrains
                    # how far we may swerve to get round something else
                    found.append((x1, y1, x2, y2))
                    if not self._in_corridor((x1 + x2) / 2 / w, y2 / h):
                        continue
                    height_frac = (y2 - y1) / h
                    if height_frac >= self.stop_height_frac:
                        stop = True
                    elif height_frac >= self.slow_height_frac:
                        slow = True
                self._stop, self._slow, self._boxes = stop, slow, found
                self._failures = 0
                self._result_time = time.monotonic()
            except Exception:
                self._failures += 1
                logger.exception("detection failed (%d consecutive)", self._failures)
                time.sleep(0.2)  # back off rather than spin hot on a hard fault
            self.fps = 1.0 / max(time.monotonic() - t0, 1e-3)

    def run_threaded(self, image):
        self.image = image
        now = time.monotonic()

        unhealthy = None
        if self._failures >= self.max_failures:
            unhealthy = f"{self._failures} consecutive inference failures"
        elif now - self._result_time > self.max_result_age:
            unhealthy = f"no detection for {now - self._result_time:.1f}s"

        if unhealthy:
            if now - self._last_stale_log > 5.0:
                logger.error("YoloGuard unhealthy (%s) — forcing stop", unhealthy)
                self._last_stale_log = now
            return True, False, False, self.fps, []

        return self._stop, self._slow, True, self.fps, self._boxes

    def shutdown(self):
        self.running = False
