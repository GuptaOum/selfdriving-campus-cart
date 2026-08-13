#!/usr/bin/env python3
"""
Export and quantize the pretrained vision models for the campus cart.

Run this on your LAPTOP or Google Colab (x86, internet), NOT on the Pi.
Outputs land in ./exported_models/ — copy that folder to the Pi
(e.g. `scp -r exported_models pi@raspberrypi.local:~/mycar/models/`).

Produces:
  exported_models/segformer_sidewalk.onnx          (fp32, ~14 MB)
  exported_models/segformer_sidewalk_int8.onnx     (INT8, ~4 MB — use this on the Pi)
  exported_models/segformer_labels.json            (id -> class name, drivable ids)
  exported_models/yolov8n_ncnn_model/              (NCNN export for the Pi)

Usage:
    pip install transformers torch onnx onnxruntime ultralytics
    python export_models.py [--skip-seg] [--skip-yolo]
"""
import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "exported_models"

SEGFORMER_MODEL_ID = "segments-tobias/segformer-b0-finetuned-segments-sidewalk"

# sidewalk-semantic label names we treat as drivable ground for the cart.
# These are the dataset's real strings — the "flat-" prefix is part of the name.
# Full list: https://huggingface.co/datasets/segments/sidewalk-semantic
# Matching is case-insensitive because some checkpoints capitalise them.
DRIVABLE_CLASS_NAMES = [
    "flat-road",
    "flat-sidewalk",
    "flat-crosswalk",
    "flat-cyclinglane",
    "flat-parkingdriveway",
    # deliberately NOT drivable:
    #   flat-curb      — the raised edge; driving onto it beaches a 1/8 cart
    #   flat-railtrack — rails
    #   nature-terrain — grass/soil shoulder; soft, the cart sinks and strands
]

# Unpaved and ambiguous ground. Campus sand/dirt paths often land in
# "void-ground" rather than any flat-* class, so if Stage 2 shows your unpaved
# stretches masked as non-drivable, try enabling this BEFORE spending a day on
# fine-tuning. It is off by default because "void" also covers genuinely
# unclear pixels, and treating those as road is optimistic.
INCLUDE_UNPAVED = False
UNPAVED_CLASS_NAMES = ["void-ground"]

SEG_INPUT_SIZE = 256  # square input on the Pi; keep in sync with myconfig.py

def export_segformer():
    import torch
    from transformers import SegformerForSemanticSegmentation

    print(f"Downloading {SEGFORMER_MODEL_ID} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    wanted = {n.lower() for n in DRIVABLE_CLASS_NAMES}
    if INCLUDE_UNPAVED:
        wanted |= {n.lower() for n in UNPAVED_CLASS_NAMES}
    drivable_ids = [i for i, name in id2label.items() if name.lower() in wanted]

    print("\nClass mapping:")
    for i, name in sorted(id2label.items()):
        print(f"  {i:3d}  {name}{'   <-- DRIVABLE' if i in drivable_ids else ''}")

    # An empty list silently produces an all-black mask, which looks exactly
    # like "the model thinks nothing is drivable" — a whole day lost to
    # debugging a typo. Fail here instead.
    if not drivable_ids:
        raise SystemExit(
            "\nERROR: no class name in DRIVABLE_CLASS_NAMES matched this "
            "checkpoint's labels (listed above). Edit DRIVABLE_CLASS_NAMES at "
            "the top of this script to use the exact strings shown.")

    labels_path = OUT_DIR / "segformer_labels.json"
    labels_path.write_text(json.dumps(
        {"id2label": id2label, "drivable_ids": drivable_ids,
         "input_size": SEG_INPUT_SIZE}, indent=2))
    print(f"\nWrote {labels_path}")
    print(f"  drivable ids: {drivable_ids}")
    print(f"  = {[id2label[i] for i in drivable_ids]}")

    onnx_path = OUT_DIR / "segformer_sidewalk.onnx"
    dummy = torch.randn(1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["pixel_values"], output_names=["logits"],
        # fixed shape: dynamic axes cost speed on the Pi and we only ever use one size
        opset_version=13,
    )
    print(f"Wrote {onnx_path}")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = OUT_DIR / "segformer_sidewalk_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"Wrote {int8_path} — this is the one the Pi runs")


def export_yolo():
    from ultralytics import YOLO

    print("Downloading YOLOv8n (COCO) ...")
    model = YOLO("yolov8n.pt")
    # NCNN is the fastest CPU backend on a Pi 4 (~9 FPS at 640, faster at 320)
    exported = model.export(format="ncnn", imgsz=320)
    print(f"NCNN export at: {exported}")
    print(f"Move the *_ncnn_model folder into {OUT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if not args.skip_seg:
        export_segformer()
    if not args.skip_yolo:
        export_yolo()
    print("\nDone. Copy exported_models/ to the Pi.")


if __name__ == "__main__":
    main()
