import cv2
import numpy as np
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Smooth a binary mask video spatially and temporally.")
    parser.add_argument("--input", default="campussample_bw_mask.mp4", help="Input binary mask video")
    parser.add_argument("--output", default="campussample_bw_mask_smoothed.mp4", help="Output side-by-side video")
    parser.add_argument("--alpha", type=float, default=0.25, help="Temporal blending factor (0.0 to 1.0). Lower = smoother but more lag.")
    parser.add_argument("--kernel-size", type=int, default=15, help="Kernel size for spatial morphology")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output will be side-by-side: [Original | Smoothed]
    out_w = w * 2
    out_h = h
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    # Kernel for spatial smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel_size, args.kernel_size))
    
    # State for temporal smoothing
    prev_mask_float = None

    print(f"Processing {total_frames} frames. Alpha: {args.alpha}, Kernel: {args.kernel_size}x{args.kernel_size}")
    
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale binary (0 or 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # ---------------------------------------------------------
        # 1. Spatial Smoothing (Morphological Operations)
        # ---------------------------------------------------------
        # Close: Fills in small holes in the white drivable area
        spatially_smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Open: Removes jagged spikes and isolated white noise in the black area
        spatially_smoothed = cv2.morphologyEx(spatially_smoothed, cv2.MORPH_OPEN, kernel)

        # ---------------------------------------------------------
        # 2. Temporal Smoothing (Exponential Moving Average)
        # ---------------------------------------------------------
        curr_mask_float = spatially_smoothed.astype(np.float32) / 255.0
        
        if prev_mask_float is None:
            prev_mask_float = curr_mask_float

        # Blend: alpha * current + (1 - alpha) * previous
        blended_float = (args.alpha * curr_mask_float) + ((1.0 - args.alpha) * prev_mask_float)
        
        # Threshold back to strict binary for the planner
        final_smoothed = (blended_float > 0.5).astype(np.uint8) * 255
        
        # Update history for next frame
        prev_mask_float = blended_float

        # ---------------------------------------------------------
        # Visualization: Side-by-Side
        # ---------------------------------------------------------
        final_color = cv2.cvtColor(final_smoothed, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(frame, "Raw BW Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(final_color, "Smoothed (Spatial + Temporal)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Horizontally stack the frames
        side_by_side = np.hstack((frame, final_color))
        writer.write(side_by_side)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    writer.release()
    
    elapsed = time.time() - start_time
    print(f"\nDone! Processed in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"Saved side-by-side comparison to {args.output}")

if __name__ == "__main__":
    main()
