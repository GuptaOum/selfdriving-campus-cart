import time
import os
import resource
import cv2
import json
import numpy as np
import onnxruntime as ort

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except:
        return None

def main():
    video_path = "campussample_trimmed.mp4"
    onnx_path = "fastscnn_selfdriving_int8.onnx"
    labels_path = "fastscnn_labels.json"

    print("=" * 55)
    print("      RASPBERRY PI 4 AUTONOMOUS VISION BENCHMARK      ")
    print("=" * 55)

    temp_start = get_cpu_temp()
    if temp_start:
        print(f"🌡️  Initial CPU Temperature : {temp_start:.1f} °C")

    # Load metadata
    meta = json.loads(open(labels_path).read())
    input_size = int(meta.get("input_size", 256))

    # Initialize ONNX session on all 4 CPU cores
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Video: {video_path} ({total_frames} frames)")
    print(f"🧠 Model: {onnx_path} (INT8 Quantized)")
    print("-" * 55)
    print("Running benchmark across all 4 CPU cores...")

    latencies = []
    t_start = time.time()
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        t0 = time.perf_counter()

        # 1. Preprocess
        img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = img.transpose(2, 0, 1)[None]

        # 2. Inference
        logits = sess.run(None, {input_name: tensor})[0]

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        processed += 1
        if processed % 50 == 0:
            current_fps = processed / (time.time() - t_start)
            print(f"  Frame {processed:3d}/{total_frames} | Recent Latency: {latencies[-1]:.1f} ms | Rolling FPS: {current_fps:.2f}")

    t_end = time.time()
    cap.release()

    total_time = t_end - t_start
    avg_fps = processed / total_time
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    temp_end = get_cpu_temp()

    print("\n" + "=" * 55)
    print("                 FINAL PERFORMANCE REPORT                 ")
    print("=" * 55)
    print(f"⏱️  Total Duration       : {total_time:.2f} seconds")
    print(f"🖼️  Frames Processed     : {processed}")
    print(f"⚡ Average Throughput   : {avg_fps:.2f} FPS")
    print(f"🎯 Average Latency      : {avg_latency:.1f} ms / frame")
    print(f"🟢 Min Latency          : {min_latency:.1f} ms")
    print(f"🔴 Max Latency          : {max_latency:.1f} ms")
    print(f"📊 95th-Percentile (p95): {p95_latency:.1f} ms")
    print(f"🧠 Peak RAM Usage       : {peak_ram_mb:.1f} MB (out of 2000 MB)")
    if temp_end:
        print(f"🌡️  Final CPU Temp       : {temp_end:.1f} °C (Δ {temp_end - temp_start:+.1f} °C)")
    print("-" * 55)
    print("🚗 Autonomous Driving Reaction Feasibility:")
    for speed_kmh in [5, 10, 15]:
        speed_ms = speed_kmh * (1000 / 3600)
        distance_per_frame = speed_ms * (avg_latency / 1000.0)
        print(f"  • At {speed_kmh:2d} km/h: Car updates path every {distance_per_frame:.2f} m ({distance_per_frame*3.28084:.1f} ft)")
    print("=" * 55)

if __name__ == "__main__":
    main()
