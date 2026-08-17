#!/usr/bin/env python3
"""
Phase 1 go/no-go bench: run the ACTUAL car code (mycar/parts) on recorded
campus footage, before anything touches the car.

Record clips walking or driving the planned routes — include every sandy or
unpaved stretch, at least one speed breaker, a junction, and people about.

    python vision_bench.py --video campus_walk.mp4 \
        --seg-model exported_models/segformer_sidewalk_int8.onnx \
        --seg-labels exported_models/segformer_labels.json \
        [--yolo-model exported_models/yolov8n_ncnn_model] \
        [--homography ground_plane.json] \
        [--out annotated.mp4] [--show]

Without --homography you get segmentation and detection only. WITH it you get
the whole stack — occupancy grid, arc planning, tracking and prediction — and
a bird's-eye panel showing every candidate arc and the chosen one.

RUN IT WITH THE PLANNER. Running the front half alone hides a whole class of
problem: on real dashcam footage the planner reported STOP on every single
frame, because the camera cannot see the ground at its own bumper and those
unimaged cells were being read as obstacles. Segmentation looked perfect
throughout. Only the planner exposed it.

Go/no-go gates:
  - mask garbage on sand/dirt      -> fine-tune ~100 campus images (Colab)
  - OOM or seg < ~1 FPS on the Pi  -> Fast-SCNN, or Pi 5 8GB
  - planner stops constantly       -> check the homography and PLANNER_LATERAL_M
  - all fine                       -> proceed to on-car integration
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# import the real car code, not a copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mycar"))
from parts.seg_pilot import SegEngine            # noqa: E402
from parts.breaker_detect import BreakerDetect    # noqa: E402
from parts.local_planner import LocalPlanner     # noqa: E402

BEV_W = 300


def peak_rss_mb():
    try:
        import resource  # Linux (the Pi) only
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except ImportError:
            return None


def draw_bev(planner, grid, scores, best, moving, height):
    """Bird's-eye panel: free space, every candidate arc, the winner."""
    g = np.zeros((planner.rows, planner.cols, 3), np.uint8)
    g[grid > 0] = (58, 78, 58)
    g[grid == 0] = (26, 26, 30)
    g = cv2.resize(g, (BEV_W, height), interpolation=cv2.INTER_NEAREST)
    sx, sy = BEV_W / planner.cols, height / planner.rows
    res = planner.res

    for steer, clear, _ in scores:
        pts, x, y, th, d = [], 0.0, 0.0, 0.0, 0.0
        ang = steer * planner.max_steer_rad
        curv = np.tan(ang) / planner.wheelbase_m if abs(ang) > 1e-4 else 0.0
        while d < clear:
            th -= curv * res
            x += res * np.cos(th)
            y += res * np.sin(th)
            d += res
            c = int((planner.lateral_m - y) / res * sx)
            r = int((planner.horizon_m - x) / res * sy)
            if 0 <= c < BEV_W and 0 <= r < height:
                pts.append((c, r))
        if len(pts) > 1:
            win = abs(steer - best) < 1e-6
            cv2.polylines(g, [np.array(pts, np.int32)], False,
                          (0, 235, 255) if win else (95, 95, 105), 3 if win else 1)

    for o in moving or []:
        c = int((planner.lateral_m - o["y"]) / res * sx)
        r = int((planner.horizon_m - o["x"]) / res * sy)
        cv2.circle(g, (c, r), 7, (60, 60, 240), -1)          # where it is now
        t_meet = min(o["x"] / max(planner.assumed_speed_ms, 0.05),
                     planner.predict_horizon_s)
        pc = int((planner.lateral_m - (o["y"] + o["vy"] * t_meet)) / res * sx)
        pr = int((planner.horizon_m - (o["x"] + o["vx"] * t_meet)) / res * sy)
        cv2.circle(g, (pc, pr), 7, (60, 170, 250), 2)        # where it will be
        cv2.arrowedLine(g, (c, r), (pc, pr), (60, 170, 250), 1, tipLength=0.25)

    cv2.circle(g, (BEV_W // 2, height - 6), 5, (255, 255, 255), -1)
    cv2.putText(g, "BIRD'S-EYE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)
    cv2.putText(g, f"{planner.horizon_m:.0f}m x {2*planner.lateral_m:.0f}m",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(g, "red=now  orange=predicted", (8, height - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 170, 250), 1)
    cv2.putText(g, "yellow = chosen arc", (8, height - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 235, 255), 1)
    return g


def annotate(frame, mask, debug, angle, throttle, clear, breaker, seg_fps,
             tracks, source, clear_m):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    mask_up = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay[mask_up > 0] = (90, 220, 120)
    out = cv2.addWeighted(frame, 0.66, overlay, 0.34, 0)

    roi = int(h * 0.4)
    band_h = (h - roi) // 5
    cv2.line(out, (0, roi), (w, roi), (255, 210, 60), 1)
    pts = []
    for b, cx_frac, _ in debug.get("centroids", []):
        y = h - b * band_h - band_h // 2
        x = int(cx_frac * w)
        pts.append((x, y))
        cv2.circle(out, (x, y), 6, (40, 40, 245), -1)
        cv2.circle(out, (x, y), 6, (255, 255, 255), 1)
    if len(pts) > 1:
        cv2.polylines(out, [np.array(pts, np.int32)], False, (40, 40, 245), 2)

    for item in tracks:
        tid, x1, y1, x2, y2 = item
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (235, 120, 60), 2)
        cv2.putText(out, f"#{tid}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 120, 60), 2)

    cv2.line(out, (w // 2, roi), (w // 2, h), (200, 200, 200), 1)
    cv2.arrowedLine(out, (w // 2, h - 26), (int(w // 2 + angle * w * 0.27), h - 84),
                    (0, 235, 255), 5, tipLength=0.3)

    cv2.rectangle(out, (0, 0), (w, 68), (18, 18, 20), -1)
    col = (110, 230, 140) if clear else (70, 70, 240)
    extra = f"   clear {clear_m:4.1f}m" if clear_m is not None else ""
    txt = (f"[{source}] steer {angle:+.2f}   throttle {throttle:.2f}{extra}   "
           f"{'DRIVING' if clear else 'STOP'}   seg {seg_fps:.1f} FPS")
    if breaker:
        txt += "   [BREAKER]"
    cv2.putText(out, txt, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(out, "green=drivable  red=band centroids  orange=tracks  "
                     "yellow=steering",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 185, 185), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seg-model", required=True)
    ap.add_argument("--seg-labels", required=True)
    ap.add_argument("--yolo-model", help="NCNN model dir; omit to skip detection")
    ap.add_argument("--homography", help="ground_plane.json — enables the planner, "
                                         "tracking and prediction")
    ap.add_argument("--out", help="write annotated video here")
    ap.add_argument("--csv", help="write per-frame steering here. The summary "
                                  "reports mean/min/max, which hides WHERE a "
                                  "spike happened and what the mask looked "
                                  "like when it did; this does not.")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (walk footage is oversampled)")
    ap.add_argument("--cart-width", type=float, default=0.28)
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--horizon", type=float, default=4.0)
    ap.add_argument("--lateral", type=float, default=2.8,
                    help="half-width of the planning window; must cover your path")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="expected cruising speed, m/s. Converts arc distance into "
                         "time so moving obstacles are checked at the moment we "
                         "would actually meet them.")
    args = ap.parse_args()

    engine = SegEngine(args.seg_model, args.seg_labels)
    breaker_part = BreakerDetect()      # the part, so distance gating applies

    yolo = None
    if args.yolo_model:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model, task="detect")

    planner = None
    if args.homography:
        planner = LocalPlanner(
            homography_path=args.homography,
            cart_width_m=args.cart_width, safety_margin_m=args.margin,
            horizon_m=args.horizon, lateral_m=args.lateral,
            assumed_speed_ms=args.speed)
        if not planner.enabled:
            sys.exit(f"could not load a homography from {args.homography}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None

    seg_times, det_times, plan_times = [], [], []
    steers, stops, breakers, n, done = [], 0, 0, 0, 0

    csv_f = csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        # bands/drivable are here because a spike is nearly always one of them
        # collapsing: a pole splitting the corridor drops a band, a bad frame
        # drops the drivable fraction. Without them a spike is just a number.
        csv_w.writerow(["frame", "t_sec", "steer", "d_steer", "throttle",
                        "clear", "bands", "drivable_frac", "breaker",
                        "clear_m", "seg_ms"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % args.every:
            continue
        done += 1

        t0 = time.monotonic()
        mask = engine.infer_mask(frame)
        seg_angle, seg_throttle, seg_clear, debug = engine.steer_from_mask(mask)
        seg_times.append(time.monotonic() - t0)

        # RGB in, and hand it the mask or roadside grass reads as paint
        breaker = breaker_part.run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mask)
        if breaker:
            breakers += 1

        tracks = []
        if yolo:
            t0 = time.monotonic()
            # track(), not predict(): ids are what make velocity knowable
            res = yolo.track(frame, imgsz=320, conf=0.4, verbose=False,
                             persist=True, tracker="bytetrack.yaml")
            det_times.append(time.monotonic() - t0)
            for box in res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tid = int(box.id.item()) if box.id is not None else -1
                tracks.append((tid, x1, y1, x2, y2))

        angle, throttle, clear, clear_m = seg_angle, seg_throttle, seg_clear, None
        source, bev = "seg", None
        if planner:
            t0 = time.monotonic()
            h0, w0 = frame.shape[:2]
            ms = mask.shape[0]
            # boxes must be in MASK coordinates, which is what the planner's
            # homography was calibrated against
            scaled = [(t, x1 * ms / w0, y1 * ms / h0, x2 * ms / w0, y2 * ms / h0)
                      for (t, x1, y1, x2, y2) in tracks]
            # VIDEO time, not wall-clock: offline we run slower than real time,
            # and wall-clock would make every tracked object look far slower
            # than it is. On the car the two coincide.
            moving = planner.update_tracks(scaled, mask.shape, done / fps)
            grid = planner.inflate(
                planner.build_grid(mask),
                planner.boxes_to_grid(scaled, mask.shape) if scaled else None)
            steer, clear_m, scores = planner.plan(
                grid, goal_bias=float(np.clip(seg_angle, -1, 1)), moving=moving)
            plan_times.append(time.monotonic() - t0)

            clear = clear_m >= planner.min_clear_m
            planner._prev_steer = steer if clear else 0.0
            depth = min(1.0, clear_m / planner.horizon_m)
            angle = steer if clear else 0.0
            throttle = (planner.throttle_creep +
                        (planner.throttle_cruise - planner.throttle_creep) * depth
                        ) if clear else 0.0
            source = "planner"
            bev = draw_bev(planner, grid, scores, steer, moving, frame.shape[0])

        if csv_w:
            csv_w.writerow([
                n, f"{n / fps:.3f}", f"{angle:+.4f}",
                f"{angle - (steers[-1] if steers else 0.0):+.4f}",
                f"{throttle:.3f}", int(bool(clear)),
                len(debug.get("centroids", [])), f"{float(mask.mean()):.4f}",
                int(bool(breaker)),
                "" if clear_m is None else f"{clear_m:.2f}",
                f"{seg_times[-1]*1000:.0f}"])

        steers.append(angle)
        if not clear:
            stops += 1

        out = annotate(frame, mask, debug, angle, throttle, clear, breaker,
                       1.0 / max(seg_times[-1], 1e-3), tracks, source, clear_m)
        if bev is not None:
            out = np.hstack([out, bev])

        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(fps / args.every, 1),
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        if args.show:
            cv2.imshow("vision_bench", out)
            if cv2.waitKey(1) == 27:  # Esc
                break

    cap.release()
    if csv_f:
        csv_f.close()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    s = np.array(steers) if steers else np.zeros(1)
    print(f"\nframes processed : {done}")
    print(f"segmentation     : {1.0/np.mean(seg_times):.2f} FPS "
          f"(mean {np.mean(seg_times)*1000:.0f} ms)")
    if det_times:
        print(f"detection+track  : {1.0/np.mean(det_times):.2f} FPS "
              f"(mean {np.mean(det_times)*1000:.0f} ms)")
    if plan_times:
        print(f"planner          : {np.mean(plan_times)*1000:.1f} ms "
              f"(geometry, not inference)")
    print(f"steering         : mean {s.mean():+.3f}  min {s.min():+.3f}  "
          f"max {s.max():+.3f}")
    print(f"frames it would STOP : {stops}/{done} "
          f"({100.0*stops/max(done,1):.0f}%)")
    print(f"speed breakers seen  : {breakers} frames")
    rss = peak_rss_mb()
    if rss:
        print(f"peak RSS         : {rss:.0f} MB (budget on a 2GB Pi: ~1200 MB)")

    print("\nGATES")
    print("  mask sane on sand/dirt in the annotated video?")
    print("  seg >= 1 FPS on the Pi, RSS under budget?")
    if plan_times:
        if stops > done * 0.3:
            print(f"  ** STOPPING {100.0*stops/max(done,1):.0f}% OF FRAMES **")
            print("     check the homography, and that --lateral covers your path")
        else:
            print("  planner keeps a path open - good")
    else:
        print("  NO PLANNER RUN. Pass --homography: segmentation alone hides")
        print("  whole classes of failure, including the camera's blind zone")
        print("  at the bumper being read as an obstacle.")


if __name__ == "__main__":
    main()
