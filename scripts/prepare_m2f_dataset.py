#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hf-id", default="facebook/mask2former-swin-large-cityscapes-panoptic")
    ap.add_argument("--drivable", default="road,sidewalk")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", default="training_data_m2f")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.hf_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.hf_id)
    model = model.cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    want = {n.strip() for n in args.drivable.split(",")}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit("Cannot open video")

    out = Path(args.out)
    img_dir = out / "images"
    mask_dir = out / "masks"
    for d in (img_dir, mask_dir): d.mkdir(parents=True, exist_ok=True)

    frame_ids = []
    n = 0
    t0 = time.monotonic()
    
    print(f"Extracting dataset using Mask2Former ({args.hf_id}) on GPU...")
    while True:
        ok, frame = cap.read()
        if not ok: break
        if n % args.every == 0:
            stem = f"frame_{n:06d}"
            
            # Predict with Mask2Former
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
            
            h, w = frame.shape[:2]
            res = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]
            seg = res["segmentation"].cpu().numpy()
            infos = res["segments_info"]
            
            drivable = np.zeros((h, w), dtype=np.uint8)
            for s in infos:
                name = id2label.get(int(s["label_id"]), "")
                if name in want:
                    drivable[seg == s["id"]] = 255
            
            # Save 256x256 image and mask
            resized_img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(drivable, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(str(img_dir / f"{stem}.png"), resized_img)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), resized_mask)
            
            frame_ids.append(stem)
            if len(frame_ids) % 50 == 0:
                print(f"  {len(frame_ids)} frames...")
        n += 1

    cap.release()
    print(f"Extracted {len(frame_ids)} frames in {time.monotonic() - t0:.1f}s")

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
            "every": args.every,
            "num_classes": 2,
            "drivable_ids": [1]
        }
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

if __name__ == "__main__":
    main()
