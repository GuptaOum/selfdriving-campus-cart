#!/usr/bin/env python3
"""
Export a Fast-SCNN model to ONNX for the Raspberry Pi.

Without --weights: exports an UNTRAINED model (for FPS benchmarking only).
With --weights:    loads trained .pth checkpoint (for real deployment).

Usage:
    .venv\\Scripts\\activate
    set PYTHONIOENCODING=utf-8
    python scripts/export_fastscnn.py --weights training_data/checkpoints/fastscnn_campus_best.pth
"""
import argparse
import json
import torch
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

from fast_scnn_arch import FastSCNN

OUT_DIR = Path(__file__).resolve().parent.parent / "exported_models" / "fastscnn"
SEG_INPUT_SIZE = 256  # keep in sync with SegPilot / myconfig.py


def main():
    ap = argparse.ArgumentParser(description="Export Fast-SCNN to ONNX")
    ap.add_argument("--weights", default=None,
                    help="path to trained .pth checkpoint (omit for untrained)")
    ap.add_argument("--num-classes", type=int, default=2,
                    help="number of output classes (default: 2 = bg + drivable)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Instantiating FastSCNN architecture...")
    model = FastSCNN(num_classes=args.num_classes)

    if args.weights:
        print(f"Loading weights from {args.weights}")
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("Trained weights loaded!")
    else:
        print("WARNING: Using random weights -- for FPS benchmarking only!")

    model.eval()

    # Labels file
    id2label = {0: "void-ground", 1: "flat-sidewalk"}
    drivable_ids = [1]

    labels_path = OUT_DIR / "fastscnn_labels.json"
    labels_path.write_text(json.dumps(
        {"id2label": id2label, "drivable_ids": drivable_ids,
         "input_size": SEG_INPUT_SIZE,
         "profile": "fastscnn_campus",
         "trained": args.weights is not None}, indent=2))
    print(f"\nWrote {labels_path}")

    # ONNX export
    out_dir = Path("../exported_models/fastscnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "fastscnn_selfdriving_fp32.onnx"
    dummy = torch.randn(1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # FastSCNN returns a tuple, we want the first element
            logits = self.model(x)[0]
            # Our BCE training only learned channel 1.
            # To make np.argmax(logits, axis=0) work, we set channel 0 to 0.0.
            # Then argmax compares channel 1 against 0.0.
            ch1 = logits[:, 1:2, :, :]
            ch0 = torch.zeros_like(ch1)
            return torch.cat([ch0, ch1], dim=1)

    wrapped_model = ExportWrapper(model)
    wrapped_model.eval()

    torch.onnx.export(
        wrapped_model, dummy, str(onnx_path),
        input_names=["pixel_values"], output_names=["logits"],
        opset_version=13,
    )
    print(f"Wrote {onnx_path}")

    # INT8 quantization
    int8_path_final = out_dir / "fastscnn_selfdriving_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path_final), weight_type=QuantType.QInt8)
    print(f"Wrote {int8_path_final} -- this is the one the Pi runs!")


if __name__ == "__main__":
    main()
