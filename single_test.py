from deepface import DeepFace
import cv2
import config
import time

camera = cv2.VideoCapture(config.CAMERA_ID)

if not camera.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera opened.")
print("Look directly at the camera.")
print("Press SPACE to capture your face.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Single Face Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:

        print("\nFace captured!")
        print("Running face recognition...")
        print("Please wait...\n")

        try:

            result = DeepFace.find(
                img_path=frame,
                db_path=config.DATABASE_PATH,
                model_name="Facenet512",
                detector_backend="retinaface",
                enforce_detection=True,
                silent=False
            )

            if len(result) > 0 and not result[0].empty:

                best = result[0].iloc[0]

                identity = best["identity"]
                distance = best["distance"]

                identity = identity.replace("\\", "/")

                name = identity.split("/")[-2]

                print("==============================")
                print("RECOGNITION SUCCESSFUL")
                print("==============================")
                print("Name:", name)
                print("Distance:", distance)
                print("Image:", identity)
                print("==============================")

            else:

                print("No matching person found.")

        except Exception as e:

            print("Recognition error:")
            print(e)

        print("\nPress SPACE to test again.")
        print("Press Q to quit.")

camera.release()
cv2.destroyAllWindows()

print("Test stopped.")