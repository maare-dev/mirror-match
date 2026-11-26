from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    model_path = Path(__file__).parent / "yolo26s-pose.pt"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)

        for result in results:
            annotated_frame = result.plot()

            cv2.imshow("YOLO Pose", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
