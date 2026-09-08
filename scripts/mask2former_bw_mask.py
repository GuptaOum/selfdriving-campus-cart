#!/usr/bin/env python3
"""
Mask2Former binary drivable mask: white = drivable, black = everything else.

Runs facebook/mask2former-swin-large-cityscapes-panoptic on every frame,
collapses the semantic labels to a binary mask (road + sidewalk = white),
and writes the result as a grayscale video.
"""
import argparse
import time

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hf-id",
                    default="facebook/mask2former-swin-large-cityscapes-panoptic")
    ap.add_argument("--drivable", default="road,sidewalk",
                    help="Comma-separated Cityscapes class names to treat as drivable")
    ap.add_argument("--crop-top", type=float, default=0.0)
    ap.add_argument("--crop-bottom", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=1,
                    help="Process every Nth frame (1 = all frames)")
    ap.add_argument("--out", required=True,
                    help="Output video path (.mp4)")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(args.hf_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.hf_id)
    model = model.cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    want = {n.strip() for n in args.drivable.split(",")}

    print(f"cuda: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"drivable classes: {want}", flush=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    n = done = 0
    times = []

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

        # Build binary mask: drivable = 255 (white), else = 0 (black)
        drivable = np.zeros((bh_px, w), dtype=np.uint8)
        for s in infos:
            name = id2label.get(int(s["label_id"]), "")
            if name in want:
                drivable[seg == s["id"]] = 255

        # Paste into full-height frame (non-band regions stay black)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1] = drivable

        # Convert to 3-channel grayscale for video writing
        out = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.every, (w, h))
        writer.write(out)

        if done % 100 == 0:
            elapsed = np.mean(times[-100:])
            print(f"  {done}/{total_frames} frames, {elapsed:.0f} ms/frame, "
                  f"{1000/elapsed:.1f} FPS", flush=True)

    cap.release()
    if writer:
        writer.release()

    print(f"\nframes processed : {done}")
    print(f"inference        : {1000 / np.mean(times):.2f} FPS "
          f"(mean {np.mean(times):.0f} ms)")
    print(f"output           : {args.out}")


if __name__ == "__main__":
    main()
