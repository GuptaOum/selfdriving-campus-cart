#!/usr/bin/env python3
"""
Phase 1 go/no-go bench: run the ACTUAL car vision code (mycar/parts) on
recorded campus footage, before anything touches the car.

Record clips walking the planned routes (include every sandy/unpaved stretch
and at least one speed breaker), then:

    python vision_bench.py --video campus_walk.mp4 \
        --seg-model exported_models/segformer_sidewalk_int8.onnx \
        --seg-labels exported_models/segformer_labels.json \
        [--yolo-model exported_models/yolov8n_ncnn_model] \
        [--out annotated.mp4] [--show]

Reports: seg FPS, det FPS, peak RSS. Writes an annotated video with the
drivable mask, band centroids, steering value, breaker flag, and detections.

Go/no-go gates (from the plan):
  - mask garbage on sand/dirt  -> fine-tune with ~100 campus images (Colab)
  - OOM or seg < ~1 FPS @256px on the Pi -> upgrade to Pi 5 8GB
  - both fine -> proceed to on-car integration
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# import the real car code, not a copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mycar"))
from parts.seg_pilot import SegEngine            # noqa: E402
from parts.breaker_detect import detect_breaker  # noqa: E402


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


def annotate(frame, mask, debug, angle, throttle, breaker, seg_fps, boxes):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    mask_up = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay[mask_up > 0] = (0, 180, 0)
    out = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    for b, cx_frac, _ in debug.get("centroids", []):
        band_h = int(h * 0.6) // 5
        y = h - b * band_h - band_h // 2
        cv2.circle(out, (int(cx_frac * w), y), 6, (0, 0, 255), -1)

    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(out, label, (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    txt = f"angle {angle:+.2f}  throttle {throttle:.2f}  seg {seg_fps:.1f} FPS"
    if breaker:
        txt += "  [BREAKER]"
    cv2.putText(out, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seg-model", required=True)
    ap.add_argument("--seg-labels", required=True)
    ap.add_argument("--yolo-model", help="NCNN model dir; omit to skip detection")
    ap.add_argument("--out", help="write annotated video here")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (walk footage is oversampled)")
    args = ap.parse_args()

    engine = SegEngine(args.seg_model, args.seg_labels)
    yolo = None
    if args.yolo_model:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model, task="detect")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    writer = None

    seg_times, det_times, n = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % args.every:
            continue

        t0 = time.monotonic()
        mask = engine.infer_mask(frame)
        angle, throttle, clear, debug = engine.steer_from_mask(mask)
        seg_times.append(time.monotonic() - t0)

        breaker, _ = detect_breaker(frame)

        boxes = []
        if yolo:
            t0 = time.monotonic()
            res = yolo.predict(frame, imgsz=320, conf=0.4, verbose=False)
            det_times.append(time.monotonic() - t0)
            for box in res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((x1, y1, x2, y2, res[0].names[int(box.cls)]))

        out = annotate(frame, mask, debug, angle, throttle, breaker,
                       1.0 / max(seg_times[-1], 1e-3), boxes)
        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                    (out.shape[1], out.shape[0]))
            writer.write(out)
        if args.show:
            cv2.imshow("vision_bench", out)
            if cv2.waitKey(1) == 27:  # Esc
                break

    cap.release()
    if writer:
        writer.release()

    print(f"\nframes processed : {len(seg_times)}")
    print(f"segmentation     : {1.0 / np.mean(seg_times):.2f} FPS "
          f"(mean {np.mean(seg_times)*1000:.0f} ms)")
    if det_times:
        print(f"detection        : {1.0 / np.mean(det_times):.2f} FPS "
              f"(mean {np.mean(det_times)*1000:.0f} ms)")
    rss = peak_rss_mb()
    if rss:
        print(f"peak RSS         : {rss:.0f} MB (budget on 2GB Pi: ~1200 MB)")
    print("\nGATES: seg >= 1 FPS on the Pi? mask sane on sand/dirt in the "
          "annotated video? RSS under budget?")


if __name__ == "__main__":
    main()
