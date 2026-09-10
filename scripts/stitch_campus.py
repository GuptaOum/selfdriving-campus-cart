import cv2
import sys

print("Stitching campus videos side-by-side...")
cap1 = cv2.VideoCapture("../vision_bench_campussample_roi0.30.mp4") # The one generated earlier with m2f (pre-trained)
cap2 = cv2.VideoCapture("../vision_bench_campussample_finetuned_roi0.30.mp4") # The one being generated now (fine-tuned)

fps = cap1.get(cv2.CAP_PROP_FPS)
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("../side_by_side_comparison_campus.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w1+w2, max(h1, h2)))

frames = 0
while True:
    ok1, f1 = cap1.read()
    ok2, f2 = cap2.read()
    if not ok1 or not ok2:
        break
    
    cv2.putText(f1, "BEFORE (Pre-Trained)", (10, h1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(f1, "BEFORE (Pre-Trained)", (10, h1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(f2, "AFTER (Fine-Tuned)", (10, h2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(f2, "AFTER (Fine-Tuned)", (10, h2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    
    combined = cv2.hconcat([f1, f2])
    out.write(combined)
    frames += 1
    if frames % 100 == 0:
        print(f"Stitched {frames} frames...")

cap1.release()
cap2.release()
out.release()
print("Done! Saved to side_by_side_comparison_campus.mp4")
