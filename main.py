import cv2, numpy as np
from ultralytics import YOLO

model = YOLO("yolo26s-pose.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model(frame, stream=True, verbose=True)
    for result in results:
        break
    break

cap.release()
cv2.destroyAllWindows()
