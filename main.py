import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time

import cv2
from deepface import DeepFace

from security_event import save_unknown_face

DB_PATH = "database"
CHECK_EVERY_N_FRAMES = 10
MOTION_MIN_AREA = 1500
MOTION_HOLD_SECONDS = 3
UNKNOWN_CHECKS_REQUIRED = 3


def detect_motion(frame, subtractor):
    mask = subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > MOTION_MIN_AREA for c in contours)


def recognize(frame):
    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=DB_PATH,
            enforce_detection=True,
            silent=True,
        )
    except ValueError:
        return None

    if len(results) > 0 and not results[0].empty:
        identity = results[0].iloc[0]["identity"]
        return os.path.basename(os.path.dirname(identity))
    return "Unknown"


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frame_count = 0
last_motion = 0.0
label = None
unknown_streak = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()

    if detect_motion(frame, subtractor):
        last_motion = now
    scanning = (now - last_motion) < MOTION_HOLD_SECONDS

    if scanning and frame_count % CHECK_EVERY_N_FRAMES == 0:
        try:
            label = recognize(frame)
        except Exception as error:
            print(f"Recognition error: {error}")
            label = None

        if label == "Unknown":
            unknown_streak += 1
            if unknown_streak >= UNKNOWN_CHECKS_REQUIRED:
                save_unknown_face(frame)
                unknown_streak = 0
        else:
            unknown_streak = 0

    if not scanning:
        label = None
        unknown_streak = 0

    display = frame.copy()
    status = "SCANNING" if scanning else "IDLE"
    cv2.putText(display, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if label == "Unknown":
        cv2.putText(display, "Unknown", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif label:
        cv2.putText(display, f"Person: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Face Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()