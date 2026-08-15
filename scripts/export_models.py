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
    pip install -r requirements-dev.txt
    python export_models.py [--skip-seg] [--skip-yolo]

On Windows, run it with PYTHONIOENCODING=utf-8 — torch's ONNX exporter prints
a unicode tick on success and the default cp1252 console cannot encode it,
which crashes the script AFTER the model has already been written.
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
# Choose with --profile. Default is footpath.
#
# WHY THERE ARE TWO. Including the road classes merges the footpath and the
# street beside it into ONE continuous corridor. The band centroids then land
# between the two and the cart steers off the path toward traffic. Measured on
# bench footage of two campuses: mean steering -0.370 on a sidewalk that should
# read ~0.0, and the same clip with no street in view came back to -0.009. The
# mask was correct both times — grass and kerbs were excluded properly. It is
# the class list that decides where the corridor ends.
DRIVABLE_PROFILES = {
    # The cart stays on pedestrian ground. Use this for any route that has a
    # footpath, which is the normal campus case.
    "footpath": [
        "flat-sidewalk",
        "flat-crosswalk",    # a footpath continues across a road at a crossing
        "flat-cyclinglane",  # campus cycle lanes are legitimate cart route
    ],
    # For campus interiors where an internal road IS the only route and there
    # is no footpath at all. Only pick this where you accept the cart sharing a
    # carriageway — and re-check steering on that route before trusting it.
    "road": [
        "flat-sidewalk",
        "flat-crosswalk",
        "flat-cyclinglane",
        "flat-road",
        "flat-parkingdriveway",
    ],
}
DEFAULT_PROFILE = "footpath"

# deliberately NOT drivable in either profile:
#   flat-curb      — the raised edge; driving onto it beaches a 1/8 cart
#   flat-railtrack — rails
#   nature-terrain — grass/soil shoulder; soft, the cart sinks and strands

# Unpaved and ambiguous ground. Campus sand/dirt paths often land in
# "void-ground" rather than any flat-* class, so if Stage 2 shows your unpaved
# stretches masked as non-drivable, try enabling this BEFORE spending a day on
# fine-tuning. It is off by default because "void" also covers genuinely
# unclear pixels, and treating those as road is optimistic.
INCLUDE_UNPAVED = False
UNPAVED_CLASS_NAMES = ["void-ground"]

SEG_INPUT_SIZE = 256  # square input on the Pi; keep in sync with myconfig.py

def export_segformer(profile=DEFAULT_PROFILE):
    import torch
    from transformers import SegformerForSemanticSegmentation

    print(f"Downloading {SEGFORMER_MODEL_ID} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    class_names = DRIVABLE_PROFILES[profile]
    print(f"\nDrivable profile: {profile} -> {class_names}")
    wanted = {n.lower() for n in class_names}
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
            f"\nERROR: no class name in profile '{profile}' matched this "
            "checkpoint's labels (listed above). Edit DRIVABLE_PROFILES at "
            "the top of this script to use the exact strings shown.")

    labels_path = OUT_DIR / "segformer_labels.json"
    # profile is recorded so a labels file on the Pi can be identified later —
    # "which ground did I export this one to accept?" is otherwise a guess.
    labels_path.write_text(json.dumps(
        {"id2label": id2label, "drivable_ids": drivable_ids,
         "input_size": SEG_INPUT_SIZE, "profile": profile}, indent=2))
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
    ap.add_argument("--profile", choices=sorted(DRIVABLE_PROFILES),
                    default=DEFAULT_PROFILE,
                    help="which ground counts as drivable; 'footpath' keeps "
                         "the cart off the road, 'road' allows carriageway "
                         "routes that have no footpath")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if not args.skip_seg:
        export_segformer(args.profile)
    if not args.skip_yolo:
        export_yolo()
    print("\nDone. Copy exported_models/ to the Pi.")


if __name__ == "__main__":
    main()
