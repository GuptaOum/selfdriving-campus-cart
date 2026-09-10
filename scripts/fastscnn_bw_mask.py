import argparse
import time
import json
import cv2
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.loads(open(args.labels).read())
    drivable_ids = np.array(meta["drivable_ids"], dtype=np.int64)
    input_size = int(meta.get("input_size", 256))

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

    done = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = img.transpose(2, 0, 1)[None]
        
        logits = sess.run(None, {input_name: tensor})[0]
        classes = np.argmax(logits[0], axis=0).astype(np.int64)
        drivable = np.isin(classes, drivable_ids).astype(np.uint8)
        
        drivable_full = cv2.resize(drivable, (w, h), interpolation=cv2.INTER_NEAREST)
        drivable_full = drivable_full * 255
        
        out_frame = cv2.cvtColor(drivable_full, cv2.COLOR_GRAY2BGR)
        writer.write(out_frame)

        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total} frames")

    cap.release()
    writer.release()
    print(f"Done. Wrote {args.out}")

if __name__ == "__main__":
    main()
