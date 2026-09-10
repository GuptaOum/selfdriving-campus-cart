import cv2
import numpy as np

img = cv2.imread('../videos/selfDRIVING_frame0.png')
h, w = img.shape[:2]

# Crop amounts from bottom
crops = [0, 100, 200, 300]
font = cv2.FONT_HERSHEY_SIMPLEX

canvases = []
for c in crops:
    crop_h = h - c
    cropped = img[:crop_h, :]
    # Resize to width 300 to fit in a grid
    scale = 300.0 / w
    new_h = int(crop_h * scale)
    resized = cv2.resize(cropped, (300, new_h))
    
    # Pad to same height (original scaled height)
    max_h = int(h * scale)
    padded = np.zeros((max_h, 300, 3), dtype=np.uint8)
    padded[:new_h, :] = resized
    
    cv2.putText(padded, f"Crop: {c}px", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(padded, f"Height: {crop_h}", (10, 60), font, 0.8, (0, 255, 0), 2)
    canvases.append(padded)

grid = np.hstack(canvases)
cv2.imwrite('C:/Users/hp/.gemini/antigravity/brain/a27b37ae-7952-4d8a-8123-cdc7573c84a5/scratch/crop_test.png', grid)
