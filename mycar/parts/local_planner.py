"""
LocalPlanner — steer AROUND obstacles instead of stopping at them.

The band-centroid steering in seg_pilot.py follows the middle of the path.
That is fine on open ground and useless when a person is standing on it: the
corridor test sees a blocked path and the cart stops, even when there is a
metre of clear tarmac to the left.

This plans instead of following. It is a Dynamic-Window-style local planner,
the same idea sidewalk delivery robots actually run, and it needs no training
data and no new neural network. The perception it needs already exists — a
person is `human-person` in the SegFormer mask, so they are already a hole in
the drivable area. What was missing was something able to reason: "that gap is
80 cm, I am 28 cm wide, I fit."

How it works:

  1. Warp the drivable mask into a BIRD'S-EYE occupancy grid using a
     one-time ground-plane homography. Reasoning about clearance in image
     pixels is hopeless because perspective makes near and far pixels mean
     different distances; in the grid, one cell is one fixed number of
     centimetres everywhere.
  2. Inflate every obstacle by the cart's half-width plus a margin. After
     inflating, the cart can be treated as a POINT — a standard trick that
     turns "does this 28 cm box fit" into "is this cell free".
  3. Roll out candidate steering angles as arcs using bicycle kinematics,
     and score each by how far it gets before hitting something, how well it
     points where we want to go, and how little it jerks the wheel.
  4. Drive the best arc.

Falls back to whatever seg_pilot decided if no homography is calibrated, so
enabling this cannot silently break a working cart.
"""
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LocalPlanner:
    """
    Non-threaded DonkeyCar part. Roughly 3-8 ms per tick — it is plain
    geometry over a small grid, not inference.

    Inputs:  seg/mask, seg/angle, seg/corridor_clear, yolo/boxes, nav/command
    Outputs: plan/angle, plan/throttle, plan/clear (bool), plan/distance_m
    """

    def __init__(self, homography_path,
                 cart_width_m=0.28, wheelbase_m=0.32, safety_margin_m=0.12,
                 horizon_m=3.0, grid_res_m=0.05, lateral_m=1.6,
                 n_candidates=21, max_steer_rad=0.52,
                 smoothness_weight=0.35, heading_weight=0.5,
                 throttle_cruise=0.30, throttle_creep=0.16,
                 min_clear_m=0.45):
        """
        :param homography_path: JSON from scripts/calibrate_ground_plane.py.
               Missing or unreadable -> planner disables itself and the cart
               keeps using seg_pilot's steering.
        :param cart_width_m: measure it, including wheels at full lock.
        :param wheelbase_m: front axle to rear axle. Sets the turning arcs.
        :param safety_margin_m: extra clearance beyond the cart's own width.
               This is what stops it shaving past a person's toes.
        :param horizon_m: how far ahead to plan. Beyond ~3 m the mask is too
               few pixels per metre to trust.
        :param max_steer_rad: physical steering limit, ~30 deg by default.
        :param min_clear_m: an arc must reach at least this far to count as
               passable. Below it, treat the way as blocked and stop.
        """
        self.cart_width_m = cart_width_m
        self.wheelbase_m = wheelbase_m
        self.safety_margin_m = safety_margin_m
        self.horizon_m = horizon_m
        self.res = grid_res_m
        self.lateral_m = lateral_m
        self.max_steer_rad = max_steer_rad
        self.smoothness_weight = smoothness_weight
        self.heading_weight = heading_weight
        self.throttle_cruise = throttle_cruise
        self.throttle_creep = throttle_creep
        self.min_clear_m = min_clear_m

        self.rows = int(horizon_m / grid_res_m)          # X forward
        self.cols = int(2 * lateral_m / grid_res_m)      # Y lateral
        self.n_candidates = n_candidates
        self._prev_steer = 0.0

        # inflating by the cart's half-width lets us treat it as a point
        inflate_cells = int(round(
            (cart_width_m / 2.0 + safety_margin_m) / grid_res_m))
        k = max(3, 2 * inflate_cells + 1)
        self._inflate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        self.homography = self._load_homography(homography_path)
        self.enabled = self.homography is not None
        if not self.enabled:
            logger.warning(
                "no ground-plane homography at %s — LocalPlanner is disabled "
                "and steering falls back to seg_pilot. Run "
                "scripts/calibrate_ground_plane.py to enable obstacle "
                "avoidance.", homography_path)
        else:
            logger.info("LocalPlanner ready: %.2f m x %.2f m grid at %.0f mm, "
                        "cart %.0f cm + %.0f cm margin",
                        horizon_m, 2 * lateral_m, grid_res_m * 1000,
                        cart_width_m * 100, safety_margin_m * 100)

    @staticmethod
    def _load_homography(path):
        try:
            data = json.loads(Path(path).read_text())
            return np.array(data["homography"], dtype=np.float32)
        except (OSError, ValueError, KeyError):
            return None

    # ---------- occupancy grid ----------

    def build_grid(self, mask):
        """
        Drivable mask (image space) -> bird's-eye free-space grid.

        Returns a uint8 array, 1 = free, 0 = blocked, with row 0 nearest the
        cart and column 0 on its far left.
        """
        # warp straight into grid resolution; the homography maps image pixels
        # to ground metres, and the scale matrix maps metres to grid cells
        scale = np.array([
            [0.0, -1.0 / self.res, self.lateral_m / self.res],
            [-1.0 / self.res, 0.0, self.horizon_m / self.res],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        m = scale @ self.homography
        grid = cv2.warpPerspective(mask, m, (self.cols, self.rows),
                                   flags=cv2.INTER_NEAREST)
        return (grid > 0).astype(np.uint8)

    def inflate(self, grid, extra_blocked=None):
        """
        Grow obstacles by the cart's half-width + margin, so collision
        checking reduces to testing single cells.
        """
        blocked = (grid == 0).astype(np.uint8)
        if extra_blocked is not None:
            blocked |= extra_blocked
        blocked = cv2.dilate(blocked, self._inflate_kernel)
        return (1 - blocked).astype(np.uint8)

    def boxes_to_grid(self, boxes, image_shape):
        """
        Project detection boxes onto the ground and mark them blocked.

        The mask usually covers a detected person already, but detection is
        the more reliable signal for a THIN obstacle — a pole or a standing
        leg can be a handful of pixels the segmentation smooths over, while
        YOLO still boxes it confidently.
        """
        extra = np.zeros((self.rows, self.cols), np.uint8)
        if not boxes:
            return extra
        h, w = image_shape[:2]
        for x1, y1, x2, y2 in boxes:
            # an object touches the ground along the BOTTOM edge of its box;
            # its top is in the air and would project to nonsense
            foot = np.array([[[(x1 + x2) / 2.0, y2]]], dtype=np.float32)
            ground = cv2.perspectiveTransform(foot, self.homography)[0][0]
            x_m, y_m = float(ground[0]), float(ground[1])
            row = int((self.horizon_m - x_m) / self.res)
            col = int((self.lateral_m - y_m) / self.res)
            if 0 <= row < self.rows and 0 <= col < self.cols:
                half = max(1, int(((x2 - x1) / w) * 0.5 / self.res))
                cv2.circle(extra, (col, row), half, 1, -1)
        return extra

    # ---------- arc rollout ----------

    def _arc_clearance(self, grid, steer):
        """
        How far the cart gets along this steering arc before hitting
        something, in metres. Bicycle model at constant curvature.

        Sign convention: steer is DonkeyCar's, where POSITIVE MEANS RIGHT. The
        ground frame has Y pointing left, so theta is subtracted rather than
        added — getting this backwards makes the planner dodge the wrong way
        round every obstacle, which looks like working code right up until it
        drives into someone.
        """
        angle = steer * self.max_steer_rad
        # a perfectly straight arc has infinite radius; guard the tangent
        curvature = math.tan(angle) / self.wheelbase_m if abs(angle) > 1e-4 else 0.0

        x = y = theta = 0.0
        step = self.res
        travelled = 0.0
        while travelled < self.horizon_m:
            theta -= curvature * step
            x += step * math.cos(theta)
            y += step * math.sin(theta)
            travelled += step

            row = int((self.horizon_m - x) / self.res)
            col = int((self.lateral_m - y) / self.res)
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                break            # left the planned area; stop counting here
            if grid[row, col] == 0:
                return travelled
        return travelled

    def plan(self, grid, goal_bias=0.0):
        """
        Score every candidate arc and return (steer, clearance_m, debug).

        :param goal_bias: -1 to +1, where we would like to head — from the GPS
               junction command, or the segmentation centreline on open path.
        """
        best, best_score, scores = 0.0, -1e9, []
        for i in range(self.n_candidates):
            steer = -1.0 + 2.0 * i / (self.n_candidates - 1)
            clear = self._arc_clearance(grid, steer)

            # progress dominates: an arc that goes nowhere is worthless however
            # nicely it points
            score = clear / self.horizon_m
            score -= self.heading_weight * abs(steer - goal_bias) / 2.0
            score -= self.smoothness_weight * abs(steer - self._prev_steer) / 2.0

            scores.append((steer, clear, score))
            if score > best_score:
                best, best_score = steer, score

        best_clear = next(c for s, c, _ in scores if s == best)
        return best, best_clear, scores

    # ---------- part interface ----------

    def run(self, mask=None, seg_angle=0.0, corridor_clear=False,
            boxes=None, nav_command=None):
        if not self.enabled or mask is None:
            # planner off or no perception: pass seg_pilot's decision through
            return seg_angle or 0.0, 0.0, bool(corridor_clear), 0.0

        try:
            grid = self.build_grid(mask)
            extra = self.boxes_to_grid(boxes, mask.shape) if boxes else None
            grid = self.inflate(grid, extra)
        except cv2.error:
            logger.exception("grid build failed — falling back to seg steering")
            return seg_angle or 0.0, 0.0, bool(corridor_clear), 0.0

        # where we would LIKE to go; the planner may overrule it to get past
        # something, which is the entire point
        goal_bias = {"LEFT": -0.6, "RIGHT": 0.6}.get(nav_command, seg_angle or 0.0)
        steer, clear_m, _ = self.plan(grid, goal_bias=max(-1.0, min(1.0, goal_bias)))

        if clear_m < self.min_clear_m:
            # every arc is blocked within a cart length; there is no way past
            self._prev_steer = 0.0
            return 0.0, 0.0, False, clear_m

        self._prev_steer = steer
        depth = min(1.0, clear_m / self.horizon_m)
        throttle = self.throttle_creep + (self.throttle_cruise - self.throttle_creep) * depth
        return steer, throttle, True, clear_m
