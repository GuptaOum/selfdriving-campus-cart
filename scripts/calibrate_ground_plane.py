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
4. Tape a FIFTH cross somewhere inside the rectangle and measure it too.
   Four points always fit a homography EXACTLY, so their error is always zero
   no matter how badly you clicked — it proves nothing. A fifth point is not
   used to build the homography, so it is the only honest way to find out
   whether the calibration is right.
5. Run this, click the four markers IN THAT SAME ORDER, then the fifth, and
   press S to save.

    python scripts/calibrate_ground_plane.py --camera 0 --verify 1.5,0
    python scripts/calibrate_ground_plane.py --image frame.jpg --verify 1.5,0

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
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 5:
        clicks.append((float(x), float(y)))


def sanity_problems(clicks, ground):
    """
    Catch the calibration mistakes that reprojection error cannot.

    With exactly four correspondences the homography fits them EXACTLY, so
    reprojection error is always ~0 — including when a marker was clicked in
    the wrong place or the four were clicked in the wrong order. That number
    is not a check, it is arithmetic, and reporting it as accuracy is worse
    than reporting nothing.

    What can be checked without extra measurements is that the picture agrees
    with the tape: a marker measured farther forward must appear higher in the
    image, and one measured further left must appear further left. That is
    what catches a swapped or misclicked corner.

    Assumes the camera is upright and looking forward — true for a fixed mast
    mount, and a rolled camera is its own problem.
    """
    problems = []
    n = len(ground)

    for i in range(n):
        for j in range(i + 1, n):
            (xi, yi), (xj, yj) = ground[i], ground[j]
            (ui, vi), (uj, vj) = clicks[i], clicks[j]

            # farther forward -> higher in the image (smaller pixel v)
            if abs(xi - xj) > 0.25 and (xi > xj) != (vi < vj):
                problems.append(
                    f"point {i+1} is {abs(xi-xj):.2f} m "
                    f"{'farther' if xi > xj else 'nearer'} than point {j+1}, "
                    f"but you clicked it {'lower' if vi > vj else 'higher'} "
                    "in the image")

            # further left -> further left in the image (smaller pixel u)
            if abs(yi - yj) > 0.25 and (yi > yj) != (ui < uj):
                problems.append(
                    f"point {i+1} is {abs(yi-yj):.2f} m "
                    f"{'left' if yi > yj else 'right'} of point {j+1}, "
                    f"but you clicked it to the "
                    f"{'right' if ui > uj else 'left'} in the image")

    # a quadrilateral that crosses itself means the click order is wrong
    pts = np.array(clicks[:4], np.float32)
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append(np.sign(np.cross(b - a, c - b)))
    if len(set(cross)) > 1:
        problems.append("the four clicks do not form a simple quadrilateral — "
                        "they were probably clicked out of order")
    return problems


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="camera index, e.g. 0")
    src.add_argument("--image", help="a still frame instead")
    ap.add_argument("--ground", help="8 comma-separated metres: "
                                     "x1,y1,x2,y2,x3,y3,x4,y4")
    ap.add_argument("--verify", metavar="X,Y",
                    help="metres of a FIFTH marker, e.g. 1.5,0. Tape one more "
                         "cross somewhere inside the rectangle, click it last, "
                         "and its error is the only honest accuracy number you "
                         "can get: four points always fit exactly, a fifth "
                         "does not have to.")
    args = ap.parse_args()

    verify_pt = None
    if args.verify:
        v = [float(n) for n in args.verify.split(",")]
        if len(v) != 2:
            raise SystemExit("--verify needs exactly 2 numbers: X,Y")
        verify_pt = (v[0], v[1])

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
        if len(clicks) >= 4:
            pts = np.array(clicks[:4], np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, (0, 220, 0), 2)

        want = 5 if verify_pt is not None else 4
        if len(clicks) < want:
            msg = (f"click the VERIFICATION marker" if len(clicks) == 4
                   else f"click point {len(clicks) + 1} of {want}")
        else:
            msg = "S to save, R to redo"
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

            # NOT a reprojection check: four points fit exactly, so that error
            # is always ~0 and would call a broken calibration perfect.
            problems = sanity_problems(clicks[:4], ground)
            if problems:
                print("\nTHESE CLICKS DISAGREE WITH YOUR MEASUREMENTS:")
                for p in problems:
                    print(f"  - {p}")
                print("Press R and click again in the order listed above, or "
                      "re-check which marker is which.")
                continue

            verify_err = None
            if verify_pt is not None:
                if len(clicks) < 5:
                    print("\nclick the fifth (verification) marker too, "
                          "then press S")
                    continue
                got = cv2.perspectiveTransform(
                    np.array([[clicks[4]]], np.float32), H)[0][0]
                verify_err = float(np.hypot(got[0] - verify_pt[0],
                                            got[1] - verify_pt[1]))
                print(f"\nVERIFICATION MARKER")
                print(f"  measured : X={verify_pt[0]:+.2f} m  Y={verify_pt[1]:+.2f} m")
                print(f"  computed : X={got[0]:+.2f} m  Y={got[1]:+.2f} m")
                print(f"  error    : {verify_err * 100:.1f} cm")
                if verify_err > 0.15:
                    print("  OVER 15 cm OUT. The planner compares gaps against a "
                          "28 cm cart, so this much error makes its clearance "
                          "maths meaningless. Re-measure and re-click.")
                elif verify_err > 0.08:
                    print("  usable, but re-clicking more precisely is worth "
                          "the two minutes")
                else:
                    print("  good")
            else:
                print("\nNo verification marker given, so this calibration is "
                      "UNCHECKED — four points always fit exactly. Tape a "
                      "fifth cross and re-run with --verify X,Y to find out "
                      "whether it is actually right.")

            OUT.write_text(json.dumps({
                "homography": H.tolist(),
                "image_points": clicks[:4],
                "ground_points": [list(g) for g in ground],
                "verify_image_point": clicks[4] if len(clicks) > 4 else None,
                "verify_ground_point": list(verify_pt) if verify_pt else None,
                "verify_error_m": verify_err,
                "image_size": [frame.shape[1], frame.shape[0]],
            }, indent=2))
            print(f"wrote {OUT} — copy it to the Pi and set PLANNER_HOMOGRAPHY")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
