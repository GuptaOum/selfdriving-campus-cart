#!/usr/bin/env python3
"""
One-time ground-plane calibration. Produces the homography the LocalPlanner
needs to reason about clearance in centimetres instead of pixels.

WHY THIS IS NEEDED
------------------
In a camera image, a pixel near the bottom is a few centimetres of ground and a
pixel near the horizon is metres. You cannot ask "is that gap wider than my
cart" of an image; the question only means something on the ground plane. This
maps one to the other.

WHAT TO DO
----------
1. Mount the camera EXACTLY where it will live and do not move it again. Move
   it later and this calibration is void — re-run it.
2. Put four markers on flat ground in front of the cart in a rectangle you
   have measured. Roughly 1 m wide by 2 m long works well. Tape crosses,
   bottle caps, anything you can see clearly.
3. Measure them in metres relative to the cart: X forward from the front axle,
   Y to the LEFT. So a 1 x 2 m rectangle starting 0.5 m ahead is
   (0.5, 0.5) (0.5, -0.5) (2.5, -0.5) (2.5, 0.5).
4. Run this, click the four markers IN THAT SAME ORDER, press S to save.

    python scripts/calibrate_ground_plane.py --camera 0
    python scripts/calibrate_ground_plane.py --image frame.jpg

Copy ground_plane.json to the Pi next to your models and set
PLANNER_HOMOGRAPHY in myconfig.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path("ground_plane.json")

# X forward, Y left, in metres. Edit to match what you actually taped down.
DEFAULT_GROUND = [
    (0.5, 0.5),    # near-left
    (0.5, -0.5),   # near-right
    (2.5, -0.5),   # far-right
    (2.5, 0.5),    # far-left
]

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((float(x), float(y)))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="camera index, e.g. 0")
    src.add_argument("--image", help="a still frame instead")
    ap.add_argument("--ground", help="8 comma-separated metres: "
                                     "x1,y1,x2,y2,x3,y3,x4,y4")
    args = ap.parse_args()

    ground = DEFAULT_GROUND
    if args.ground:
        v = [float(n) for n in args.ground.split(",")]
        if len(v) != 8:
            raise SystemExit("--ground needs exactly 8 numbers")
        ground = list(zip(v[0::2], v[1::2]))

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        cap = None
    else:
        cap = cv2.VideoCapture(args.camera)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("cannot read from camera")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    print(__doc__)
    print("Click these ground points in order:")
    for i, (x, y) in enumerate(ground):
        print(f"  {i + 1}. X={x:+.2f} m forward, Y={y:+.2f} m left")
    print("\nSPACE = grab a fresh frame   R = reset clicks   "
          "S = save   Q = quit")

    while True:
        if cap is not None and not clicks:
            ok, live = cap.read()
            if ok:
                frame = live

        view = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (int(x) + 9, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(clicks) == 4:
            pts = np.array(clicks, np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, (0, 220, 0), 2)

        msg = (f"click point {len(clicks) + 1} of 4" if len(clicks) < 4
               else "S to save, R to redo")
        cv2.putText(view, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            clicks.clear()
        if key == ord(' ') and cap is not None:
            clicks.clear()
        if key == ord('s'):
            if len(clicks) != 4:
                print("click all four points first")
                continue
            H, _ = cv2.findHomography(np.array(clicks, np.float32),
                                      np.array(ground, np.float32))
            if H is None:
                print("homography failed — are the four points collinear?")
                continue

            # sanity check: map the clicks back and see how far off we land
            back = cv2.perspectiveTransform(
                np.array([clicks], np.float32), H)[0]
            err = [float(np.hypot(*(b - np.array(g))))
                   for b, g in zip(back, ground)]
            worst = max(err)
            print(f"\nreprojection error: worst {worst * 100:.1f} cm, "
                  f"mean {sum(err) / 4 * 100:.1f} cm")
            if worst > 0.15:
                print("WARNING: over 15 cm out. Re-measure the rectangle and "
                      "click the markers more precisely — the planner's "
                      "clearance maths is only as good as this.")

            OUT.write_text(json.dumps({
                "homography": H.tolist(),
                "image_points": clicks,
                "ground_points": [list(g) for g in ground],
                "reprojection_error_m": err,
                "image_size": [frame.shape[1], frame.shape[0]],
            }, indent=2))
            print(f"wrote {OUT} — copy it to the Pi and set PLANNER_HOMOGRAPHY")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
