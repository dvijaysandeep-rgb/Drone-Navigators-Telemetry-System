import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.6

def process_drone_feed(source=0):
    cap = cv2.VideoCapture(source)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)

        screen_width = frame.shape[1]
        left_bound = screen_width * 0.4
        right_bound = screen_width * 0.6

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                label = model.names[cls]

                center_x = (x1 + x2) / 2

                if left_bound < center_x < right_bound:
                    color = (0, 0, 255)
                    status = "TARGET LOCK"
                else:
                    color = (0, 255, 0)
                    status = "SCANNING"

                box_height = y2 - y1

                if box_height > 0:
                    distance = round(1000 / box_height, 2)
                else:
                    distance = 0

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {distance} m",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame,
            f"Drone HUD | FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.line(
            frame,
            (int(left_bound), 0),
            (int(left_bound), frame.shape[0]),
            (255, 0, 0),
            1
        )

        cv2.line(
            frame,
            (int(right_bound), 0),
            (int(right_bound), frame.shape[0]),
            (255, 0, 0),
            1
        )

        cv2.imshow("Drone Navigator Telemetry System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_drone_feed(0)