# AI-Smart-Surveillance
# AI-Powered Smart Surveillance System

A real-time home security system that uses face recognition and motion detection to identify unauthorized intruders and send instant alerts with a photo to the owner's phone through Telegram.

## Features

- Real-time webcam monitoring
- Motion detection (background subtraction), so recognition only runs when something moves
- Face recognition with DeepFace to tell authorized people from unknown faces
- Automatic snapshot of every unknown person, saved with a timestamp
- Security log of all events
- Instant Telegram alert with the intruder's photo
- Cooldown between alerts to avoid spam

## How It Works

1. The webcam feed is read frame by frame.
2. The motion detector checks for movement. The system switches to SCANNING while there is motion and goes back to IDLE when it stops.
3. While scanning, DeepFace compares the detected face with the registered faces in the `database/` folder.
4. A known face is labelled with the person's name.
5. If an unknown face is seen in several consecutive checks, the system saves a snapshot, writes a log entry and sends a Telegram alert.

## Tech Stack

- Python
- OpenCV
- DeepFace (TensorFlow)
- Telegram Bot API

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Main program: camera loop, motion detection, recognition and alerts |
| `security_event.py` | Saves snapshots, writes the log and sends the Telegram alert |
| `motion_detector.py` | Motion detection |
| `fast_recognize.py` | Fast face recognition |
| `register.py` | Register authorized faces |
| `build_database.py`, `database.py` | Face database |
| `config.py` | Settings |
| `camera_test.py`, `live_test.py`, `single_test.py`, `test_*.py` | Test scripts |

## Setup

1. Clone the repository:
```bash
git clone https://github.com/prathameshdoiphode377-spec/AI-Smart-Surveillance.git
cd AI-Smart-Surveillance
```

2. Create a virtual environment and install the requirements:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install requests
```

3. Add authorized people. Put their photos in `database/<person_name>/`, for example `database/Prathamesh/1.jpg`.

4. Create a Telegram bot with @BotFather and get your token. Get your chat ID from @userinfobot. Send your bot a message once so it can message you.

5. Set the credentials in the terminal:
```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

## Usage

```bash
python main.py
```

Press `q` in the camera window to quit.

## Output

- Unknown face snapshots are saved in `unknown_faces/`.
- Events are logged in `security_logs/events.txt`.
- A photo alert arrives in your Telegram chat.

## Settings

These values are at the top of `main.py` and `security_event.py`:

| Setting | Meaning |
|---|---|
| `CHECK_EVERY_N_FRAMES` | How often recognition runs |
| `MOTION_MIN_AREA` | Minimum movement size that counts as motion |
| `UNKNOWN_CHECKS_REQUIRED` | Unknown results needed before an alert |
| `COOLDOWN_SECONDS` | Minimum time between two alerts |

## Privacy Note

The face database, intruder photos and logs are excluded from the repository through `.gitignore`. Never commit your bot token.
