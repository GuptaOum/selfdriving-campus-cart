"""
YoloGuard — pretrained COCO person/obstacle detector as a safety layer.

YOLOv8n exported to NCNN (see scripts/export_models.py). It does NOT steer;
it only classifies "person (or large obstacle) in my corridor" into
slow / stop signals for the SafetyArbiter. HC-SR04 remains the last line —
this just reacts sooner and at range.
"""
import logging
import time

logger = logging.getLogger(__name__)

# COCO classes that should stop a 1/8-scale cart when in the corridor
STOP_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle", 15: "cat",
                16: "dog", 2: "car"}


class YoloGuard:
    """
    Threaded DonkeyCar part.

    Inputs:  cam/image_array (RGB)
    Outputs: yolo/stop (bool), yolo/slow (bool), yolo/fps

    Corridor test: bottom-center point of a detection must fall inside a
    trapezoid spanning [corridor_bottom_frac] of the width at the bottom of the
    frame, narrowing toward the horizon. bbox height fraction is the range
    proxy: tall box = close = stop, medium = slow.
    """

    def __init__(self, model_path, imgsz=320, conf=0.4,
                 corridor_bottom_frac=0.75, horizon_frac=0.45,
                 stop_height_frac=0.45, slow_height_frac=0.22):
        from ultralytics import YOLO  # NCNN backend loads via ultralytics
        self.model = YOLO(str(model_path), task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.corridor_bottom_frac = corridor_bottom_frac
        self.horizon_frac = horizon_frac
        self.stop_height_frac = stop_height_frac
        self.slow_height_frac = slow_height_frac

        self.image = None
        self.stop = False
        self.slow = False
        self.fps = 0.0
        self.running = True
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
            try:
                results = self.model.predict(img, imgsz=self.imgsz,
                                             conf=self.conf, verbose=False)
                h, w = img.shape[:2]
                for box in results[0].boxes:
                    if int(box.cls) not in STOP_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if not self._in_corridor((x1 + x2) / 2 / w, y2 / h):
                        continue
                    height_frac = (y2 - y1) / h
                    if height_frac >= self.stop_height_frac:
                        stop = True
                    elif height_frac >= self.slow_height_frac:
                        slow = True
            except Exception:
                logger.exception("detection failed (fail-safe: no stop signal, "
                                 "sonar still covers)")
            self.stop, self.slow = stop, slow
            self.fps = 1.0 / max(time.monotonic() - t0, 1e-3)

    def run_threaded(self, image):
        self.image = image
        return self.stop, self.slow, self.fps

    def shutdown(self):
        self.running = False
