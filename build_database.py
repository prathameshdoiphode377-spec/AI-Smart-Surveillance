from deepface import DeepFace
import os
import pickle
import config

DATABASE_PATH = config.DATABASE_PATH
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"

embeddings = []

print("Building face database...")
print("Database:", DATABASE_PATH)

for person in os.listdir(DATABASE_PATH):

    person_path = os.path.join(DATABASE_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing: {person}")

    for filename in os.listdir(person_path):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_path, filename)

        try:

            representation = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=True
            )

            if len(representation) > 0:

                embeddings.append({
                    "name": person,
                    "image": image_path,
                    "embedding": representation[0]["embedding"]
                })

                print(f"  Added: {filename}")

        except Exception as e:

            print(f"  Failed: {filename}")
            print(f"  Error: {e}")


with open("face_database.pkl", "wb") as file:

    pickle.dump(embeddings, file)


print("\n==============================")
print("DATABASE BUILD COMPLETE")
print("==============================")
print("Total face images:", len(embeddings))
print("Saved as: face_database.pkl")