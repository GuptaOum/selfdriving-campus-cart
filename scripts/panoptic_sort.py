#!/usr/bin/env python3
"""
Mask2Former panoptic + SORT (Kalman + Hungarian), Cityscapes palette.

What changed from the greedy IoU version, which produced 665 ids in 900
frames:

  * Each track carries a constant-velocity Kalman filter, so it PREDICTS
    where the object should be and coasts through frames where panoptic
    drops the detection. That is what stops one car becoming six ids.
  * Association is global (Hungarian) rather than greedy first-match, and is
    forbidden across classes.
  * min_area is raised so distant parking-lot clutter never enters tracking.
    That clutter, not the near traffic, generated most of the churn.
  * A track must survive min_hits frames before it is drawn, so a one-frame
    false positive never gets an id.

No perspective grid. Top of frame is left uncropped; only the hood is cut.
Steering still comes from SegEngine.steer_from_mask on full-height geometry.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "mycar"))
from parts.seg_pilot import SegEngine   # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:                      # greedy fallback
    HAVE_SCIPY = False

CITYSCAPES = {
    "road": (100, 190, 255), "sidewalk": (244, 35, 232),  # road recoloured light blue
    "_road_cityscapes": (128, 64, 128),
    "building": (70, 70, 70), "wall": (102, 102, 156),
    "fence": (190, 153, 153), "pole": (153, 153, 153),
    "traffic light": (250, 170, 30), "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35), "terrain": (152, 251, 152),
    "sky": (70, 130, 180), "person": (220, 20, 60), "rider": (255, 0, 0),
    "car": (0, 0, 142), "truck": (0, 0, 70), "bus": (0, 60, 100),
    "train": (0, 80, 100), "motorcycle": (0, 0, 230), "bicycle": (119, 11, 32),
}
THINGS = {"car", "truck", "bus", "train", "motorcycle", "bicycle",
          "person", "rider"}


def bgr(name, default=(200, 200, 200)):
    r, g, b = CITYSCAPES.get(name, default)
    return (b, g, r)


def to_z(b):
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect]."""
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w * h,
                     w / float(h + 1e-6)]).reshape(4, 1)


def to_bbox(x):
    """[cx, cy, area, aspect, ...] -> [x1,y1,x2,y2]."""
    w = np.sqrt(max(x[2, 0], 1.0) * max(x[3, 0], 1e-6))
    h = max(x[2, 0], 1.0) / max(w, 1e-6)
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0,
                     x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


class KalmanBox:
    """Constant-velocity box filter - the SORT formulation."""
    count = 0

    def __init__(self, bbox, label, score):
        self.F = np.eye(7)
        for i, j in ((0, 4), (1, 5), (2, 6)):
            self.F[i, j] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0          # velocities start very uncertain
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0            # area/aspect are the noisy parts

        self.x = np.zeros((7, 1))
        self.x[:4] = to_z(bbox)

        KalmanBox.count += 1
        self.id = KalmanBox.count
        self.label, self.score = label, score
        self.time_since_update, self.hits, self.age = 0, 1, 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return to_bbox(self.x)

    def update(self, bbox, score):
        z = to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.score = score

    @property
    def bbox(self):
        return to_bbox(self.x)


def iou_matrix(dets, trks):
    m = np.zeros((len(dets), len(trks)), np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(trks):
            xx0, yy0 = max(d[0], t[0]), max(d[1], t[1])
            xx1, yy1 = min(d[2], t[2]), min(d[3], t[3])
            if xx1 <= xx0 or yy1 <= yy0:
                continue
            inter = (xx1 - xx0) * (yy1 - yy0)
            a = (d[2] - d[0]) * (d[3] - d[1])
            b = (t[2] - t[0]) * (t[3] - t[1])
            m[i, j] = inter / (a + b - inter)
    return m


class Sort:
    def __init__(self, iou_thresh=0.25, max_age=15, min_hits=3):
        self.iou_thresh, self.max_age, self.min_hits = iou_thresh, max_age, min_hits
        self.tracks = []

    def update(self, dets):
        """dets: list of (label, x1, y1, x2, y2, score)."""
        preds = [t.predict() for t in self.tracks]
        boxes = [d[1:5] for d in dets]
        labels = [d[0] for d in dets]

        matched, un_det = {}, set(range(len(dets)))
        if boxes and preds:
            M = iou_matrix(boxes, preds)
            for i in range(len(boxes)):        # never associate across classes
                for j, t in enumerate(self.tracks):
                    if labels[i] != t.label:
                        M[i, j] = 0.0
            if HAVE_SCIPY:
                rr, cc = linear_sum_assignment(-M)
                pairs = list(zip(rr, cc))
            else:
                pairs, usedc = [], set()
                for i in np.argsort(-M.max(axis=1)):
                    j = int(np.argmax(M[i]))
                    if j not in usedc:
                        pairs.append((i, j))
                        usedc.add(j)
            for i, j in pairs:
                if M[i, j] >= self.iou_thresh:
                    matched[j] = i
                    un_det.discard(i)

        for j, t in enumerate(self.tracks):
            if j in matched:
                i = matched[j]
                t.update(np.array(boxes[i], float), dets[i][5])

        for i in sorted(un_det):
            self.tracks.append(KalmanBox(np.array(boxes[i], float),
                                         labels[i], dets[i][5]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [(t.id, t.label, t.bbox, t.score) for t in self.tracks
                if t.time_since_update == 0 and
                (t.hits >= self.min_hits or t.age <= self.min_hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hf-id",
                    default="facebook/mask2former-swin-large-cityscapes-panoptic")
    ap.add_argument("--seg-model",
                    default="exported_models/segformer_sidewalk_int8.onnx")
    ap.add_argument("--seg-labels",
                    default="exported_models/segformer_labels_road.json")
    ap.add_argument("--drivable", default="road")
    ap.add_argument("--crop-top", type=float, default=0.0)
    ap.add_argument("--crop-bottom", type=float, default=0.34)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--min-area", type=int, default=6000)
    ap.add_argument("--min-box-h-frac", type=float, default=0.08,
                    help="box height as a fraction of the inferred band height. "
                         "A distance proxy: far vehicles are short in frame. "
                         "Anything smaller is segmented but never tracked.")
    ap.add_argument("--min-bottom-frac", type=float, default=0.40,
                    help="box bottom must sit below this fraction of the band, "
                         "which drops everything sitting near the horizon.")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--no-hud", action="store_true", default=False,
                    help="Do not draw the top telemetry HUD bar")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained(args.hf_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.hf_id)
    model = model.cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    want = {n.strip() for n in args.drivable.split(",")}
    print(f"cuda: {torch.cuda.get_device_name(0)}   scipy: {HAVE_SCIPY}", flush=True)
    print(f"min_area {args.min_area}  crop_top {args.crop_top}  "
          f"crop_bottom {args.crop_bottom}", flush=True)

    steer_engine = SegEngine(args.seg_model, args.seg_labels, crop_bottom=0.0)
    sort = Sort()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    times, steers, rows, counts = [], [], [], []
    stops = n = done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % args.every:
            continue
        done += 1
        h, w = frame.shape[:2]
        y0 = max(0, int(round(h * args.crop_top)))
        y1 = max(y0 + 1, int(round(h * (1.0 - args.crop_bottom))))
        band = frame[y0:y1]
        bh_px = y1 - y0

        t0 = time.monotonic()
        rgb = cv2.cvtColor(band, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        res = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(bh_px, w)])[0]
        seg = res["segmentation"].cpu().numpy()
        infos = res["segments_info"]
        ms = (time.monotonic() - t0) * 1000.0
        times.append(ms)

        colour = np.zeros((bh_px, w, 3), np.uint8)
        drivable = np.zeros((bh_px, w), np.uint8)
        dets = []
        for s in infos:
            name = id2label.get(int(s["label_id"]), "")
            m = (seg == s["id"])
            colour[m] = bgr(name)
            if name in want:
                drivable |= m.astype(np.uint8)
            if name in THINGS:
                ys, xs = np.nonzero(m)
                if xs.size < args.min_area:
                    continue
                bx0, bx1 = int(xs.min()), int(xs.max())
                by0, by1 = int(ys.min()), int(ys.max())
                # near-field gate: a far vehicle is short and sits high in the
                # band. Tracking those is what filled the frame with boxes.
                if (by1 - by0) < args.min_box_h_frac * bh_px:
                    continue
                if by1 < args.min_bottom_frac * bh_px:
                    continue
                dets.append((name, bx0, by0 + y0, bx1, by1 + y0,
                             float(s.get("score", 1.0))))

        tracked = sort.update(dets)

        full = np.zeros((y1, w), np.uint8)
        full[y0:y1] = drivable
        small = cv2.resize(full, (256, 256), interpolation=cv2.INTER_NEAREST)
        steer, throttle, clear, _ = steer_engine.steer_from_mask(small)
        steers.append(steer)
        stops += (not clear)
        counts.append(len(tracked))

        out = frame.copy()
        if y0:
            out[:y0] = (out[:y0] * 0.35).astype(np.uint8)
        out[y1:] = (out[y1:] * 0.35).astype(np.uint8)
        out[y0:y1] = cv2.addWeighted(out[y0:y1], 1.0 - args.alpha, colour,
                                     args.alpha, 0)

        for (tid, name, bb, score) in tracked:
            x1i, y1i, x2i, y2i = [int(v) for v in bb]
            col = bgr(name)
            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), col, 2, cv2.LINE_AA)
            tag = f"{name} #{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1i, max(0, y1i - th - 8)),
                          (x1i + tw + 8, y1i), col, -1)
            cv2.putText(out, tag, (x1i + 4, max(11, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        if args.crop_bottom > 0:
            cv2.line(out, (0, y1), (w, y1), (120, 120, 255), 2)
            cv2.putText(out, "hood - cropped before inference", (12, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 1, cv2.LINE_AA)

        if not args.no_hud:
            cv2.rectangle(out, (0, 0), (w, 74), (16, 16, 18), -1)
            col = (110, 230, 140) if clear else (70, 70, 240)
            cv2.putText(out, f"mask2former-panoptic + SORT   steer {steer:+.2f}   "
                             f"tracks {len(tracked)}   "
                             f"{'DRIVING' if clear else 'STOP'}   "
                             f"{1000 / max(ms, 1e-3):.1f} FPS",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, col, 2, cv2.LINE_AA)
            cv2.putText(out, "road light blue   only near vehicles are tracked; far ones "
                             "stay segmented but unboxed",
                        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180),
                        1, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.every, (out.shape[1], out.shape[0]))
        writer.write(out)
        rows.append([n, n / fps, f"{steer:.4f}", f"{throttle:.3f}", int(clear),
                     f"{drivable.mean():.4f}", len(tracked), f"{ms:.1f}"])
        if done % 100 == 0:
            print(f"  {done} frames, {np.mean(times[-100:]):.0f} ms, "
                  f"{np.mean(counts[-100:]):.1f} tracks/frame", flush=True)

    cap.release()
    if writer:
        writer.release()
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["frame", "t_sec", "steer", "throttle", "clear",
                         "drivable_frac", "tracks", "ms"])
            cw.writerows(rows)

    s, c = np.array(steers), np.array(counts)
    print(f"\nframes processed : {done}")
    print(f"inference        : {1000 / np.mean(times):.2f} FPS "
          f"(mean {np.mean(times):.0f} ms)")
    print(f"steering         : mean {s.mean():+.3f}  min {s.min():+.3f}  "
          f"max {s.max():+.3f}")
    print(f"tracks/frame     : mean {c.mean():.2f}  max {c.max()}")
    print(f"unique track ids : {KalmanBox.count}")
    print(f"frames it would STOP : {stops}/{done} "
          f"({100 * stops / max(done, 1):.0f}%)")


if __name__ == "__main__":
    main()
