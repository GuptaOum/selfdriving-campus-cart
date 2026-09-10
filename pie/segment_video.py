import argparse
import json
import cv2
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="campussample_trimmed.mp4", help="Video file to segment")
    ap.add_argument("--onnx", default="fastscnn_selfdriving_int8.onnx", help="ONNX model path")
    ap.add_argument("--labels", default="fastscnn_labels.json", help="Labels JSON path")
    ap.add_argument("--out", default="segmented_output.mp4", help="Output video path")
    args = ap.parse_args()

    # Load metadata
    meta = json.loads(open(args.labels).read())
    drivable_ids = np.array(meta["drivable_ids"], dtype=np.int64)
    input_size = int(meta.get("input_size", 256))

    # Initialize ONNX session
    so = ort.SessionOptions()
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

    print(f"Processing {total} frames...")
    done = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # Preprocess
        img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = img.transpose(2, 0, 1)[None]
        
        # Inference
        logits = sess.run(None, {input_name: tensor})[0]
        
        # Postprocess
        classes = np.argmax(logits[0], axis=0).astype(np.int64)
        drivable = np.isin(classes, drivable_ids).astype(np.uint8)
        
        # Upscale mask to original video size
        mask_up = cv2.resize(drivable, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Draw a beautiful green transparent overlay over the drivable area
        overlay = frame.copy()
        overlay[mask_up > 0] = (90, 220, 120)  # BGR color for light green
        
        # Blend the overlay with the original frame (34% opacity for the green)
        out_frame = cv2.addWeighted(frame, 0.66, overlay, 0.34, 0)
        
        writer.write(out_frame)

        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total} frames")

    cap.release()
    writer.release()
    print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    main()
