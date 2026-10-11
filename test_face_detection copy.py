from deepface import DeepFace
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )

        for face in faces:
            area = face["facial_area"]

            x = area["x"]
            y = area["y"]
            w = area["w"]
            h = area["h"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print(e)

    cv2.imshow("DeepFace Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()