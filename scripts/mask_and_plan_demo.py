"""
Visualise the two halves of the Phase 2 stack on a single frame:

  SegFormer    -> binary drivable mask   (perception)
  LocalPlanner -> N candidate arcs, one winner  (geometry)

Panels: input overlay | binary mask | bird's-eye grid with the rollout.

The homography here is SYNTHESISED from assumed camera geometry (height,
FOV, and the horizon row) so the picture can be drawn before the cart
exists. Metres in the bird's-eye panel are therefore NOTIONAL. Once
calibrate_ground_plane.py has been run against a taped rectangle, pass
--homography and the same code produces real distances.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycar.parts.seg_pilot import SegEngine            # noqa: E402
from mycar.parts.local_planner import LocalPlanner     # noqa: E402

PROFILES = {
    "footpath": ["flat-sidewalk", "flat-crosswalk", "flat-cyclinglane"],
    "road": ["flat-road", "flat-sidewalk", "flat-crosswalk",
             "flat-cyclinglane", "flat-parkingdriveway"],
}


def synth_homography(w, h_img, cam_h_m, fov_deg, vp_row_frac):
    """
    Pinhole camera on a flat ground plane -> image->ground homography.

    Ground frame matches LocalPlanner: X forward, Y left, both in metres.
    Pitch is recovered from where the horizon sits, which is the one thing
    you can read straight off a photo without measuring anything.
    """
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = w / 2.0, h_img / 2.0
    pitch = math.atan2(cy - vp_row_frac * h_img, fx)   # +ve = tilted down
    sp, cp = math.sin(pitch), math.cos(pitch)

    # four ground points, projected forward, then fit the inverse
    ground = [(2.0, -1.5), (2.0, 1.5), (5.0, -1.5), (5.0, 1.5)]
    img = []
    for X, Y in ground:
        depth = X * cp + cam_h_m * sp
        u = cx + fx * (-Y) / depth
        v = cy + fx * (-X * sp + cam_h_m * cp) / depth
        img.append((u, v))
    H, _ = cv2.findHomography(np.float32(img), np.float32(ground))
    return H, math.degrees(pitch)


def draw_arcs(canvas, planner, scores, best, px_per_cell):
    """Roll each candidate out again, in grid pixels, and paint it."""
    for steer, clear, _ in scores:
        angle = steer * planner.max_steer_rad
        curv = math.tan(angle) / planner.wheelbase_m if abs(angle) > 1e-4 else 0.0
        x = y = theta = 0.0
        pts = []
        travelled = 0.0
        while travelled < clear:
            theta -= curv * planner.res
            x += planner.res * math.cos(theta)
            y += planner.res * math.sin(theta)
            travelled += planner.res
            row = int((planner.horizon_m - x) / planner.res)
            col = int((planner.lateral_m - y) / planner.res)
            pts.append((int(col * px_per_cell), int(row * px_per_cell)))
        if len(pts) < 2:
            continue
        winner = abs(steer - best) < 1e-6
        cv2.polylines(canvas, [np.int32(pts)], False,
                      (0, 230, 255) if winner else (90, 90, 90),
                      3 if winner else 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--profile", default="footpath", choices=list(PROFILES))
    ap.add_argument("--seg-model", default="exported_models/segformer_sidewalk_int8.onnx")
    ap.add_argument("--seg-labels", default="exported_models/segformer_labels.json")
    ap.add_argument("--homography", help="real ground_plane.json; overrides the synthetic one")
    ap.add_argument("--cam-height", type=float, default=0.45)
    ap.add_argument("--fov", type=float, default=78.0)
    ap.add_argument("--vp", type=float, default=0.5,
                    help="horizon row as a fraction of image height")
    ap.add_argument("--crop-bottom", type=float, default=0.0,
                    help="fraction of image height dropped from the bottom "
                         "before inference — use it when the cart's own "
                         "bodywork is in frame")
    ap.add_argument("--lateral", type=float, default=2.8)
    ap.add_argument("--horizon", type=float, default=4.0)
    ap.add_argument("--command", default=None, choices=["LEFT", "RIGHT", "STRAIGHT"])
    ap.add_argument("--out", default="mask_and_plan.png")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit("could not read " + args.image)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("could not read that frame")
    else:
        sys.exit("need --image or --video")

    frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
    H_img, W_img = frame.shape[:2]

    # ---- perception -------------------------------------------------
    engine = SegEngine(args.seg_model, args.seg_labels,
                       crop_bottom=args.crop_bottom)
    meta = json.loads(Path(args.seg_labels).read_text())
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    wanted = PROFILES[args.profile]
    engine.drivable_ids = np.array(
        [i for i, n in id2label.items() if n in wanted], dtype=np.int64)

    # infer_mask crops internally, so hand it the FULL frame and keep the
    # cropped one only for display and for sizing the mask back up.
    mask_sq = engine.infer_mask(frame)                       # 256x256
    frame = engine.crop(frame)          # the cropped view IS the canonical one
    H_img, W_img = frame.shape[:2]
    mask = cv2.resize(mask_sq, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    coverage = 100.0 * float(mask.mean())

    # ---- geometry ---------------------------------------------------
    pitch = float("nan")
    if args.homography:
        hpath = args.homography
    else:
        Hm, pitch = synth_homography(W_img, H_img, args.cam_height, args.fov, args.vp)
        tmp = Path(tempfile.gettempdir()) / "_demo_homography.json"
        tmp.write_text(json.dumps({"homography": Hm.tolist()}))
        hpath = str(tmp)

    planner = LocalPlanner(hpath, lateral_m=args.lateral, horizon_m=args.horizon)
    grid = planner.build_grid(mask)
    free = planner.inflate(grid)
    bias = {"LEFT": -0.6, "RIGHT": 0.6}.get(args.command, 0.0)
    steer, clear, scores = planner.plan(free, goal_bias=bias)

    # ---- render -----------------------------------------------------
    overlay = frame.copy()
    sel = mask > 0
    overlay[sel] = (0.45 * overlay[sel]
                    + 0.55 * np.array([0, 255, 120])).astype(np.uint8)

    binary = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    px = max(1, int(H_img / planner.rows))
    bev = np.zeros((planner.rows * px, planner.cols * px, 3), np.uint8)
    big_free = cv2.resize(free * 255, (planner.cols * px, planner.rows * px),
                          interpolation=cv2.INTER_NEAREST)
    raw = cv2.resize(grid * 255, (planner.cols * px, planner.rows * px),
                     interpolation=cv2.INTER_NEAREST)
    bev[raw > 0] = (38, 62, 38)             # drivable before inflation
    bev[big_free > 0] = (58, 122, 58)       # what the cart may actually enter
    draw_arcs(bev, planner, scores, steer, px)
    bev = cv2.resize(bev, (int(bev.shape[1] * H_img / bev.shape[0]), H_img))

    def label(img, txt, y=22):
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    label(overlay, "1. SegFormer drivable (%s) %.1f%%" % (args.profile, coverage))
    label(binary, "2. binary mask (white = drivable)")
    label(bev, "3. bird's-eye + %d arcs" % planner.n_candidates)
    label(bev, "steer %+.3f  clear %.2f m" % (steer, clear), 44)

    out = np.hstack([overlay, binary, bev])
    cv2.imwrite(args.out, out)
    print("profile          : %s -> ids %s" % (args.profile, engine.drivable_ids.tolist()))
    print("drivable coverage: %.1f%% of frame" % coverage)
    if not args.homography:
        print("synthetic pitch  : %+.1f deg (NOT calibrated - metres are notional)" % pitch)
    print("grid             : %dx%d cells @ %.0f mm" % (planner.rows, planner.cols,
                                                        planner.res * 1000))
    print("chosen steer     : %+.3f   clearance %.2f m" % (steer, clear))
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
