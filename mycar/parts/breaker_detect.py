"""
BreakerDetect — classical-CV speed-breaker detector. Zero ML, zero training.

Indian speed breakers are painted in alternating yellow/black (or white/black)
stripes. HSV-threshold the yellow, require it to sit in the lower (near) part
of the frame, and require a periodic bright/dark alternation along the yellow
row band. On detection the cart enters BREAKER mode: creep throttle, steer
straight (cross perpendicular), and the segmentation "non-drivable blob"
objection is overridden for the crossing duration.

Pure detection logic is module-level so scripts/vision_bench.py reuses it.
"""
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV range for road-marking yellow (tuned wide; sun-faded paint is desaturated)
YELLOW_LO = np.array([15, 70, 90], dtype=np.uint8)
YELLOW_HI = np.array([40, 255, 255], dtype=np.uint8)


def detect_breaker(frame_bgr, roi_top=0.5, min_yellow_frac=0.04,
                   min_stripes=3):
    """
    :param frame_bgr: input frame
    :param roi_top: only look below this height fraction (breakers are on the
                    ground near us; distant ones don't matter yet)
    :param min_yellow_frac: yellow pixel fraction of ROI to consider at all
    :param min_stripes: minimum bright/dark alternations across the band
    :return: (detected: bool, band_y_frac: float) — band_y_frac is where in the
             frame the breaker sits (1.0 = at our bumper), for creep timing.
    """
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * roi_top):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

    if yellow.mean() / 255.0 < min_yellow_frac:
        return False, 0.0

    # find the row band with the most yellow — that's the breaker's image row
    row_sums = yellow.sum(axis=1)
    peak_row = int(np.argmax(row_sums))
    band = yellow[max(0, peak_row - 4):peak_row + 5, :]  # ~9 px tall band
    if band.size == 0 or band.shape[1] < 8:
        return False, 0.0
    profile = (band.mean(axis=0) > 127).astype(np.int8)

    # count yellow<->not-yellow alternations, ignoring tiny noise runs
    transitions, run = 0, 0
    prev = profile[0]
    for v in profile[1:]:
        run += 1
        if v != prev:
            if run >= max(4, w // 80):  # stripes are wide; noise runs aren't
                transitions += 1
            run = 0
            prev = v
    stripe_pairs = transitions // 2

    if stripe_pairs < min_stripes:
        return False, 0.0

    band_y_frac = (int(h * roi_top) + peak_row) / h
    return True, band_y_frac


class BreakerDetect:
    """
    Non-threaded DonkeyCar part (cheap: ~1-2 ms).

    Inputs:  cam/image_array (RGB)
    Outputs: breaker/active (bool — cart should be in creep-and-straight mode)

    Two distance thresholds, because a breaker visible far down the path is not
    a reason to creep the whole way to it:

      band_y >= approach_y_frac -> creep now, we are nearly on it
      band_y >= near_y_frac     -> it is at the bumper; latch creep for
                                   crossing_secs so we keep going while the
                                   wheels are on it and the camera can no
                                   longer see the stripes.
    """

    def __init__(self, approach_y_frac=0.62, near_y_frac=0.85,
                 crossing_secs=3.0, roi_top=0.5):
        self.approach_y_frac = approach_y_frac
        self.near_y_frac = near_y_frac
        self.crossing_secs = crossing_secs
        self.roi_top = roi_top
        self.crossing_until = 0.0

    def run(self, image):
        now = time.monotonic()
        if image is None:
            return now < self.crossing_until

        detected, band_y = detect_breaker(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), roi_top=self.roi_top)

        if detected and band_y >= self.near_y_frac:
            # at the bumper: the stripes leave the frame before the wheels are
            # across, so latch rather than relying on continued detection
            self.crossing_until = now + self.crossing_secs

        approaching = detected and band_y >= self.approach_y_frac
        return approaching or now < self.crossing_until
