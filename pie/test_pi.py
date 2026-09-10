import time
import cv2
import json
import numpy as np
import onnxruntime as ort

video_path = "campussample_trimmed.mp4"
onnx_path = "fastscnn_selfdriving_int8.onnx"
labels_path = "fastscnn_labels.json"

print(f"Loading {onnx_path}...")
meta = json.loads(open(labels_path).read())
input_size = int(meta.get("input_size", 256))

so = ort.SessionOptions()
so.intra_op_num_threads = 4  # Utilize all 4 CPU cores on the Raspberry Pi 4
sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print(f"Opening {video_path}...")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Benchmarking inference on {total_frames} frames. Please wait...")
t0 = time.time()
processed = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    # 1. Preprocess
    img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    tensor = img.transpose(2, 0, 1)[None]
    
    # 2. Run Inference
    logits = sess.run(None, {input_name: tensor})[0]
    
    processed += 1
    if processed % 50 == 0:
        print(f"  ...processed {processed}/{total_frames} frames")

t1 = time.time()
cap.release()

fps = processed / (t1 - t0)
print("\n" + "="*40)
print("BENCHMARK RESULTS (64-BIT PI 4)")
print("="*40)
print(f"Total time : {t1-t0:.2f} seconds")
print(f"Frames run : {processed}")
print(f"Average FPS: {fps:.2f} FPS")
print("="*40)
