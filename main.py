import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def main():
    model_path = Path(__file__).parent / "yolo26s-pose.pt"
    model = YOLO(model_path)
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

if __name__ == "__main__":
    main()