"""
Compose the single-frame results into one shareable 4x3 contact sheet.

Rows are frames, columns are stages: drivable overlay, the binary mask the
planner consumes, and the bird's-eye grid with every candidate arc and the
winner. Each row carries the numbers it produced, so the sheet stands on its
own without the surrounding write-up.

    python scripts/make_results_grid.py --out docs/results/summary_grid.png
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycar.parts.seg_pilot import SegEngine                       # noqa: E402
from mycar.parts.local_planner import LocalPlanner                # noqa: E402
from scripts.mask_and_plan_demo import (synth_homography,         # noqa: E402
                                        draw_arcs, PROFILES)

PW, PH = 480, 270                 # one panel
GAP = 4                           # hairline between panels, nothing else

BG = (18, 18, 20)
FG = (240, 240, 240)
HI = (0, 210, 255)

ROWS = [
    dict(key="curve_nocrop", title="Dashcam curve", sub="bonnet IN frame",
         image="curve.png", profile="road", crop=0.0, vp=0.50,
         lateral=3.5, horizon=4.0, note="model calls the tarmac vehicle-car"),
    dict(key="curve_crop", title="Dashcam curve", sub="bonnet CROPPED 16%",
         image="curve.png", profile="road", crop=0.16, vp=0.50,
         lateral=3.5, horizon=4.0, note="same weights, nothing fine-tuned"),
    dict(key="highway", title="Coastal highway", sub="no bodywork in frame",
         image="highway.png", profile="road", crop=0.0, vp=0.62,
         lateral=3.5, horizon=4.0, note="straight road, straight plan"),
    dict(key="campus", title="Campus footpath", sub="footpath profile",
         video="illinois_path_310_410.mp4", frame=400, profile="footpath",
         crop=0.0, vp=0.60, lateral=2.8, horizon=4.0,
         note="grass rejected on both sides"),
]


def text(img, s, xy, scale=0.5, col=FG, weight=1):
    # Hershey fonts are ASCII only — anything else draws as "?", which is how
    # em-dashes turned the column headers into "1 ??? drivable overlay".
    s = (s.replace("—", "-").replace("–", "-")
          .replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    s = s.encode("ascii", "replace").decode("ascii")
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, weight,
                cv2.LINE_AA)


def build_row(spec, src_dir, model, labels):
    if spec.get("video"):
        cap = cv2.VideoCapture(str(Path(src_dir) / spec["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, spec["frame"])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("cannot read " + spec["video"])
    else:
        frame = cv2.imread(str(Path(src_dir) / spec["image"]))
        if frame is None:
            raise SystemExit("cannot read " + spec["image"])
    frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

    engine = SegEngine(model, labels, crop_bottom=spec["crop"])
    meta = json.loads(Path(labels).read_text())
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    wanted = PROFILES[spec["profile"]]
    engine.drivable_ids = np.array(
        [i for i, n in id2label.items() if n in wanted], dtype=np.int64)

    mask_sq = engine.infer_mask(frame)
    frame = engine.crop(frame)
    h, w = frame.shape[:2]
    mask = cv2.resize(mask_sq, (w, h), interpolation=cv2.INTER_NEAREST)
    coverage = 100.0 * float(mask.mean())

    Hm, _ = synth_homography(w, h, 0.45, 78.0, spec["vp"])
    tmp = Path(tempfile.gettempdir()) / ("_grid_%s.json" % spec["key"])
    tmp.write_text(json.dumps({"homography": Hm.tolist(),
                               "crop_bottom": spec["crop"]}))
    planner = LocalPlanner(str(tmp), lateral_m=spec["lateral"],
                           horizon_m=spec["horizon"], crop_bottom=spec["crop"])
    grid = planner.build_grid(mask)
    free = planner.inflate(grid)
    steer, clear, scores = planner.plan(free, goal_bias=0.0)

    overlay = frame.copy()
    sel = mask > 0
    overlay[sel] = (0.45 * overlay[sel]
                    + 0.55 * np.array([0, 255, 120])).astype(np.uint8)
    binary = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    px = max(1, int(h / planner.rows))
    bev = np.zeros((planner.rows * px, planner.cols * px, 3), np.uint8)
    big = cv2.resize(free * 255, (planner.cols * px, planner.rows * px),
                     interpolation=cv2.INTER_NEAREST)
    raw = cv2.resize(grid * 255, (planner.cols * px, planner.rows * px),
                     interpolation=cv2.INTER_NEAREST)
    bev[raw > 0] = (38, 62, 38)
    bev[big > 0] = (58, 122, 58)
    draw_arcs(bev, planner, scores, steer, px)

    panels = [cv2.resize(p, (PW, PH), interpolation=cv2.INTER_AREA)
              for p in (overlay, binary, bev)]
    return panels, coverage, steer, clear, planner.n_candidates


def stamp(panel, s, corner="tl", col=FG, weight=1, scale=0.46):
    """Small caption burned into the panel itself — no chrome around the grid."""
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    x = 10 if corner[1] == "l" else panel.shape[1] - tw - 10
    y = 10 + th if corner[0] == "t" else panel.shape[0] - 10
    cv2.rectangle(panel, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
    text(panel, s, (x, y), scale, col, weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="directory holding curve.png / highway.png / the clip")
    ap.add_argument("--seg-model",
                    default="exported_models/segformer_sidewalk_int8.onnx")
    ap.add_argument("--seg-labels",
                    default="exported_models/segformer_labels.json")
    ap.add_argument("--out", default="docs/results/summary_grid.png")
    args = ap.parse_args()
    src = args.src or "."

    W = 3 * PW + 2 * GAP
    H = 4 * PH + 3 * GAP
    sheet = np.full((H, W, 3), BG, np.uint8)

    stats = []
    for r, spec in enumerate(ROWS):
        panels, cov, steer, clear, _ = build_row(spec, src, args.seg_model,
                                                 args.seg_labels)
        stamp(panels[0], spec["sub"], "tl", HI if spec["crop"] else FG,
              2 if spec["crop"] else 1)
        stamp(panels[1], "drivable %.1f%%" % cov, "tl")
        stamp(panels[2], "steer %+.3f" % steer, "tl", HI, 2)
        stamp(panels[2], "%.2f m" % clear, "bl")
        for c, p in enumerate(panels):
            y, x = r * (PH + GAP), c * (PW + GAP)
            sheet[y:y + PH, x:x + PW] = p
        stats.append((spec, cov, steer, clear))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print("wrote %s  (%dx%d)" % (args.out, W, H))
    for spec, cov, steer, clear in stats:
        print("  %-16s %-22s drivable %5.1f%%  steer %+.3f  clear %.2f m"
              % (spec["title"], spec["sub"], cov, steer, clear))


if __name__ == "__main__":
    main()
