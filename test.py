import cv2
cap = cv2.VideoCapture(0)  # Or another index (1, 2, 3)
if cap.isOpened():
    print("Camera is open")
else:
    print("Failed to open camera")
cap.release()