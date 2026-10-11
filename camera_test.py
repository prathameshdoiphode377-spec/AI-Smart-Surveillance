from deepface import DeepFace
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera test started.")
print("Look at the camera.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    try:

        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )

        print("Faces detected:", len(faces))

        for face in faces:

            area = face["facial_area"]

            x = area["x"]
            y = area["y"]
            w = area["w"]
            h = area["h"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    except Exception as e:

        print("Detection error:", e)

    cv2.imshow("Camera Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera test stopped.")