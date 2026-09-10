#!/usr/bin/env python3
"""
Prepare a distillation dataset: extract frames from a campus video and
generate pseudo-labels using the existing Segformer ONNX model.

Runs LOCALLY on your laptop — no GPU needed. Uses the INT8 or fp32
Segformer ONNX that export_models.py already produced.

Usage:
    .venv\\Scripts\\activate   (Windows)
    python scripts/prepare_distillation_dataset.py \\
        --video videos/campussample_trimmed.mp4 \\
        --seg-model exported_models/segformer/segformer_sidewalk.onnx \\
        --seg-labels exported_models/segformer/segformer_labels.json \\
        [--every 3] [--out training_data] [--val-split 0.15]

Produces:
    training_data/
        images/        frame_000000.png, frame_000003.png, …
        masks/         frame_000000.png, frame_000003.png, …
        logits/        frame_000000.npy, frame_000003.npy, …  (soft targets)
        manifest.json  { "train": [...], "val": [...], "meta": {...} }
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Segformer inference (copied from seg_pilot.py to be self-contained) ──────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_session(onnx_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def infer(sess, input_name, frame_bgr, input_size, drivable_ids):
    """Run Segformer and return (binary_mask, raw_logits)."""
    s = input_size
    img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    tensor = img.transpose(2, 0, 1)[None]

    logits = sess.run(None, {input_name: tensor})[0]          # (1, C, H/4, W/4)
    classes = np.argmax(logits[0], axis=0).astype(np.int64)    # (H/4, W/4)
    drivable = np.isin(classes, drivable_ids).astype(np.uint8)
    # Upsample to input_size
    mask = cv2.resize(drivable, (s, s), interpolation=cv2.INTER_NEAREST)
    return mask, logits[0]   # logits shape: (C, H/4, W/4)


def main():
    ap = argparse.ArgumentParser(
        description="Extract frames + Segformer pseudo-labels for Fast-SCNN training")
    ap.add_argument("--video", required=True,
                    help="path to campus video, e.g. videos/campussample_trimmed.mp4")
    ap.add_argument("--seg-model", required=True,
                    help="Segformer ONNX model (fp32 preferred for richer logits)")
    ap.add_argument("--seg-labels", required=True,
                    help="segformer_labels.json from export_models.py")
    ap.add_argument("--every", type=int, default=3,
                    help="extract every Nth frame (default: 3)")
    ap.add_argument("--out", default="training_data",
                    help="output directory (default: training_data)")
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="fraction of frames for validation (default: 0.15)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for train/val split")
    args = ap.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    meta = json.loads(Path(args.seg_labels).read_text())
    drivable_ids = np.array(meta["drivable_ids"], dtype=np.int64)
    input_size = int(meta.get("input_size", 256))
    print(f"Segformer model : {args.seg_model}")
    print(f"Drivable IDs    : {list(drivable_ids)}")
    print(f"Input size      : {input_size}×{input_size}")

    sess, input_name = make_session(args.seg_model)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video           : {args.video} ({total_frames} frames, {fps:.1f} fps)")
    print(f"Sampling        : every {args.every} frame -> ~{total_frames // args.every} samples")

    # ── Prepare output dirs ───────────────────────────────────────────────
    out = Path(args.out)
    img_dir = out / "images"
    mask_dir = out / "masks"
    logit_dir = out / "logits"
    for d in (img_dir, mask_dir, logit_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Extract + infer ───────────────────────────────────────────────────
    frame_ids = []
    n = 0
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % args.every == 0:
            stem = f"frame_{n:06d}"
            mask, logits = infer(sess, input_name, frame, input_size, drivable_ids)

            # Save resized frame at model input size (256×256)
            resized = cv2.resize(frame, (input_size, input_size),
                                 interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(img_dir / f"{stem}.png"), resized)

            # Save binary mask (0/255 for visual inspection, 0/1 for training)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask * 255)

            # Save raw logits as numpy for soft distillation
            np.save(str(logit_dir / f"{stem}.npy"), logits.astype(np.float16))

            frame_ids.append(stem)

            if len(frame_ids) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"  {len(frame_ids)} frames extracted "
                      f"({elapsed:.1f}s, {len(frame_ids)/elapsed:.1f} fps)")
        n += 1

    cap.release()
    elapsed = time.monotonic() - t0
    print(f"\nExtracted {len(frame_ids)} frames in {elapsed:.1f}s")

    # ── Train/val split ───────────────────────────────────────────────────
    random.seed(args.seed)
    indices = list(range(len(frame_ids)))
    random.shuffle(indices)
    n_val = max(1, int(len(frame_ids) * args.val_split))
    val_ids = sorted([frame_ids[i] for i in indices[:n_val]])
    train_ids = sorted([frame_ids[i] for i in indices[n_val:]])

    manifest = {
        "train": train_ids,
        "val": val_ids,
        "meta": {
            "video": args.video,
            "seg_model": args.seg_model,
            "seg_labels": args.seg_labels,
            "every": args.every,
            "input_size": input_size,
            "num_classes": 2,
            "drivable_ids": list(drivable_ids.tolist()),
            "total_frames": len(frame_ids),
        }
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Wrote {manifest_path}")
    print(f"\nDone! Copy '{out}/' to your EC2 instance for training:")
    print(f"  scp -r {out} ubuntu@<ec2-host>:~/")


if __name__ == "__main__":
    main()
