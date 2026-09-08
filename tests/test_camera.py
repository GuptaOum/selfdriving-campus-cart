import cv2


# --- Change these based on results above ---
INDEX   = 1
BACKEND = cv2.CAP_DSHOW
# -------------------------------------------

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    print(f"Failed to open camera index {INDEX}")
    exit()

print(f"Opened index {INDEX}. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    cv2.imshow(f"Camera {INDEX}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
