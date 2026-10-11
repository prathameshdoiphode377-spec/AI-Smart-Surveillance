import os
import time
import threading
from datetime import datetime

import cv2
import requests

UNKNOWN_FOLDER = "unknown_faces"
LOG_FILE = "security_logs/events.txt"
COOLDOWN_SECONDS = 30

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_last_event_time = 0.0


def _send_telegram_alert(message, image_path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured - alert not sent.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                files={"photo": photo},
                timeout=15,
            )
        if response.status_code == 200:
            print("Instant alert sent to Telegram.")
        else:
            print(f"Telegram error {response.status_code}: {response.text}")
    except requests.RequestException as error:
        print(f"Telegram alert failed: {error}")


def save_unknown_face(frame):
    global _last_event_time

    current_time = time.time()
    if current_time - _last_event_time < COOLDOWN_SECONDS:
        return False
    _last_event_time = current_time

    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
    os.makedirs("security_logs", exist_ok=True)

    now = datetime.now()
    filename = now.strftime("unknown_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(UNKNOWN_FOLDER, filename)

    cv2.imwrite(path, frame)

    with open(LOG_FILE, "a") as file:
        file.write(
            f"{now.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"UNKNOWN PERSON | "
            f"{path}\n"
        )

    print(f"Security event recorded: {path}")

    message = (
        "SECURITY ALERT: Unauthorized person detected at "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    threading.Thread(
        target=_send_telegram_alert,
        args=(message, path),
        daemon=True,
    ).start()

    return True