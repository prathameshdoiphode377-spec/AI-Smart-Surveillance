import cv2
import os
import time
import config

TOTAL_IMAGES = 20
CAPTURE_INTERVAL = 1

name = input("Enter your name: ").strip()

folder = os.path.join(config.DATABASE_PATH, name)
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam")
    exit()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])

if count >= TOTAL_IMAGES:
    print(f"{name} already has {count} images. Delete the folder to register again.")
    camera.release()
    exit()

last_capture = time.time()

print("Automatic capture started...")
print("Look at the camera and slowly turn your head.")
print(f"The program will stop after {TOTAL_IMAGES} images. Press ESC to stop early.")

while count < TOTAL_IMAGES:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

    display = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) == 1:
        status = f"Face found - saved {count}/{TOTAL_IMAGES}"
        color = (0, 255, 0)
    elif len(faces) == 0:
        status = "No face detected - look at the camera"
        color = (0, 0, 255)
    else:
        status = "Multiple faces - only one person please"
        color = (0, 165, 255)

    cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Face Registration", display)

    if len(faces) == 1 and time.time() - last_capture >= CAPTURE_INTERVAL:

        count += 1

        filename = os.path.join(folder, f"{count}.jpg")

        cv2.imwrite(filename, frame)

        print(f"Saved {count}/{TOTAL_IMAGES}")

        last_capture = time.time()

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()

print("Registration completed!")