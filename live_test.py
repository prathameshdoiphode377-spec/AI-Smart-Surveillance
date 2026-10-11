from deepface import DeepFace
import cv2
import config

camera = cv2.VideoCapture(config.CAMERA_ID)

if not camera.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Live face recognition test started.")
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

        for face in faces:

            area = face["facial_area"]

            x = area["x"]
            y = area["y"]
            w = area["w"]
            h = area["h"]

            # Make sure coordinates stay inside the image
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            result = DeepFace.find(
                img_path=face_crop,
                db_path=config.DATABASE_PATH,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )

            name = "Unknown"

            if len(result) > 0 and not result[0].empty:

                best = result[0].iloc[0]

                identity = best["identity"]
                distance = best["distance"]

                identity = identity.replace("\\", "/")

                name = identity.split("/")[-2]

                print(
                    f"Match: {name} | "
                    f"Distance: {distance:.4f}"
                )

            if name == "Unknown":

                rectangle_color = (0, 0, 255)

            else:

                rectangle_color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                rectangle_color,
                2
            )

            cv2.putText(
                frame,
                name,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                rectangle_color,
                2
            )

    except Exception as e:

        print("Recognition error:", e)

    cv2.imshow(
        "Live Face Recognition Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Test stopped.")