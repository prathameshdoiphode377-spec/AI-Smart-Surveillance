import cv2
import os
import time
import config

name = input("Enter your name: ").strip()

folder = os.path.join(config.DATABASE_PATH, name)
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam")
    exit()

count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])

last_capture = time.time()

print("Automatic capture started...")
print("Look at the camera and slowly turn your head.")
print("The program will stop after 20 images.")

while count < 20:

    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("Face Registration", frame)

    if time.time() - last_capture >= 1:

        count += 1

        filename = os.path.join(folder, f"{count}.jpg")

        cv2.imwrite(filename, frame)

        print(f"Saved {count}/20")

        last_capture = time.time()

    if cv2.waitKey(1) == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()

print("Registration completed!")