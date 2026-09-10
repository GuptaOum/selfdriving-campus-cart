import argparse
import json
import cv2
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="campussample_trimmed.mp4", help="Input video path")
    ap.add_argument("--onnx", default="fastscnn_selfdriving_int8.onnx", help="ONNX model path")
    ap.add_argument("--labels", default="fastscnn_labels.json", help="Labels JSON path")
    ap.add_argument("--out", default="bw_mask_output.mp4", help="Output B&W video path")
    args = ap.parse_args()

    meta = json.loads(open(args.labels).read())
    drivable_ids = np.array(meta["drivable_ids"], dtype=np.int64)
    input_size = int(meta.get("input_size", 256))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4  # All 4 CPU cores
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Generating Black & White mask video for {total} frames...")
    done = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # 1. Preprocess
        img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = img.transpose(2, 0, 1)[None]

        # 2. Inference
        logits = sess.run(None, {input_name: tensor})[0]
        classes = np.argmax(logits[0], axis=0).astype(np.int64)
        drivable = np.isin(classes, drivable_ids).astype(np.uint8)

        # 3. Upscale to original video resolution
        drivable_full = cv2.resize(drivable, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 4. Pure White (255) for road, Pure Black (0) for non-road
        bw_mask = (drivable_full * 255).astype(np.uint8)
        
        # Convert 1-channel grayscale to 3-channel BGR for standard MP4 encoding
        out_frame = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)
        writer.write(out_frame)

        done += 1
        if done % 50 == 0:
            print(f"  ...processed {done}/{total} frames")

    cap.release()
    writer.release()
    print(f"\nSUCCESS! Pure Black & White mask video saved to: {args.out}")

if __name__ == "__main__":
    main()
