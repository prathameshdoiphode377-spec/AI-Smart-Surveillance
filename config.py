import os

# Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder Paths
DATABASE_PATH = os.path.join(BASE_DIR, "database")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance")
MODELS_PATH = os.path.join(BASE_DIR, "models")
IMAGES_PATH = os.path.join(BASE_DIR, "images")

# Camera
CAMERA_ID = 0

# Face Detection
DETECTION_BACKEND = "retinaface"

# Face Recognition Model
MODEL_NAME = "Facenet512"

# Recognition Threshold
THRESHOLD = 0.40