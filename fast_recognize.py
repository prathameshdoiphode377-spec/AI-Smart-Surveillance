from deepface import DeepFace
import cv2
import config
import pickle
import numpy as np
import time

from attendance import create_database, mark_attendance
from security_event import save_unknown_face
from motion_detector import MotionDetector


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Facenet512"

DISTANCE_THRESHOLD = 0.35

RECOGNITION_INTERVAL = 1.0

UNKNOWN_SAVE_INTERVAL = 10


# ============================================================
# INITIALIZE MOTION DETECTOR
# ============================================================

motion_detector = MotionDetector()


# ============================================================
# LOAD FACE DATABASE
# ============================================================

print("Loading face database...")

try:
    with open("face_database.pkl", "rb") as file:
        database = pickle.load(file)

except FileNotFoundError:
    print("ERROR: face_database.pkl not found.")
    print("Run: python build_database.py")
    exit()


print(f"Loaded {len(database)} face images.")


# ============================================================
# CREATE ATTENDANCE DATABASE
# ============================================================

create_database()


# ============================================================
# VARIABLES
# ============================================================

last_marked = set()

last_recognition_time = 0

last_unknown_save = 0

motion_detected = False


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(config.CAMERA_ID)

if not camera.isOpened():

    print("ERROR: Cannot open webcam.")

    exit()


# ============================================================
# START MESSAGE
# ============================================================

print()

print("==============================")
print("AI SMART SURVEILLANCE")
print("==============================")

print("System started.")

print("Known person  = GREEN")
print("Unknown       = RED")

print("Press Q to quit.")

print("==============================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Cannot read camera.")

        break


    # --------------------------------------------------------
    # MOTION DETECTION
    # --------------------------------------------------------

    motion_detected = motion_detector.detect(frame)


    # --------------------------------------------------------
    # CURRENT TIME
    # --------------------------------------------------------

    current_time = time.time()


    # --------------------------------------------------------
    # FACE RECOGNITION
    # --------------------------------------------------------

    if (
        motion_detected
        and
        current_time - last_recognition_time
        >= RECOGNITION_INTERVAL
    ):

        try:

            # ------------------------------------------------
            # DETECT FACES
            # ------------------------------------------------

            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="retinaface",
                enforce_detection=False
            )


            # ------------------------------------------------
            # PROCESS EACH FACE
            # ------------------------------------------------

            for face in faces:

                area = face["facial_area"]

                x = area["x"]
                y = area["y"]
                w = area["w"]
                h = area["h"]


                # ------------------------------------------------
                # SAFE COORDINATES
                # ------------------------------------------------

                x1 = max(0, x)
                y1 = max(0, y)

                x2 = min(
                    frame.shape[1],
                    x + w
                )

                y2 = min(
                    frame.shape[0],
                    y + h
                )


                # ------------------------------------------------
                # CROP FACE
                # ------------------------------------------------

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if face_crop.size == 0:

                    continue


                # ------------------------------------------------
                # CREATE FACE EMBEDDING
                # ------------------------------------------------

                representation = DeepFace.represent(
                    img_path=face_crop,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )


                if len(representation) == 0:

                    continue


                query_embedding = np.array(
                    representation[0]["embedding"]
                )


                # ------------------------------------------------
                # FIND BEST MATCH
                # ------------------------------------------------

                best_name = "Unknown"

                best_distance = float("inf")


                for person in database:

                    stored_embedding = np.array(
                        person["embedding"]
                    )


                    denominator = (
                        np.linalg.norm(query_embedding)
                        *
                        np.linalg.norm(stored_embedding)
                    )


                    if denominator == 0:

                        continue


                    cosine_distance = 1 - (
                        np.dot(
                            query_embedding,
                            stored_embedding
                        )
                        /
                        denominator
                    )


                    if cosine_distance < best_distance:

                        best_distance = cosine_distance

                        best_name = person["name"]


                # ------------------------------------------------
                # MATCH DECISION
                # ------------------------------------------------

                if best_distance <= DISTANCE_THRESHOLD:

                    # --------------------------------------------
                    # KNOWN PERSON
                    # --------------------------------------------

                    name = best_name

                    box_color = (0, 255, 0)

                    label = (
                        f"{name} "
                        f"{best_distance:.2f}"
                    )


                    # --------------------------------------------
                    # ATTENDANCE
                    # --------------------------------------------

                    if name not in last_marked:

                        mark_attendance(name)

                        last_marked.add(name)

                        print(
                            f"Recognized: {name} | "
                            f"Distance: "
                            f"{best_distance:.4f}"
                        )


                else:

                    # --------------------------------------------
                    # UNKNOWN PERSON
                    # --------------------------------------------

                    name = "UNKNOWN"

                    box_color = (0, 0, 255)

                    label = (
                        f"UNKNOWN "
                        f"{best_distance:.2f}"
                    )


                    # --------------------------------------------
                    # SECURITY EVENT
                    # --------------------------------------------

                    if (
                        current_time
                        -
                        last_unknown_save
                        >= UNKNOWN_SAVE_INTERVAL
                    ):

                        save_unknown_face(frame)

                        last_unknown_save = current_time


                        print(
                            "Unknown person detected."
                        )


                # ------------------------------------------------
                # DRAW FACE RECTANGLE
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )


                # ------------------------------------------------
                # DRAW NAME
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(30, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )


            # ------------------------------------------------
            # UPDATE RECOGNITION TIMER
            # ------------------------------------------------

            last_recognition_time = current_time


        except Exception as e:

            print(
                "Recognition error:",
                e
            )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AI SMART SURVEILLANCE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MOTION STATUS
    # ========================================================

    if motion_detected:

        motion_text = "MOTION DETECTED"

    else:

        motion_text = "NO MOTION"


    cv2.putText(
        frame,
        motion_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "AI Smart Surveillance",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()

print("Surveillance stopped.")