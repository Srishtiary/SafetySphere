import cv2
import numpy as np
import time
import requests
import threading
from flask import Flask, Response
from tensorflow.keras.models import load_model

# ---------- Windows GPS for accurate location ----------
try:
    import asyncio
    import winsdk.windows.devices.geolocation as wdg
    HAVE_WINSK = True
except Exception:
    HAVE_WINSK = False


# ==============================
# TELEGRAM BOT DETAILS
# ==============================

BOT_TOKEN = "8724269811:AAHAQjND1fQoVMRsI6u1kI92wLXkjK43XoE"
CHAT_ID = "5805551664"


# ==============================
# LOAD MODELS
# ==============================

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ==============================
# TELEGRAM FUNCTIONS
# ==============================

def send_snapshot():

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    files = {"photo": open("alert.jpg", "rb")}

    data = {
        "chat_id": CHAT_ID,
        "caption": "Sad emotion detected. Live camera available."
    }

    requests.post(url, files=files, data=data)


def send_live_link():

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": "Live Camera Feed:\nhttp://172.20.10.2:5000/video"
    }

    requests.post(url, data=data)


# ==============================
# WINDOWS GPS LOCATION FUNCTION
# ==============================

async def get_windows_location():

    locator = wdg.Geolocator()

    pos = await locator.get_geoposition_async()

    lat = pos.coordinate.point.position.latitude
    lon = pos.coordinate.point.position.longitude

    return lat, lon


def send_live_location():

    if not HAVE_WINSK:
        print("Windows GPS not available")
        return

    try:

        lat, lon = asyncio.run(get_windows_location())

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendLocation"

        data = {
            "chat_id": CHAT_ID,
            "latitude": lat,
            "longitude": lon
        }

        requests.post(url, data=data)

        print("Location sent")

    except Exception as e:

        print("Location error:", e)


# ==============================
# CAMERA
# ==============================

camera = cv2.VideoCapture(0)

camera.set(3,640)
camera.set(4,480)

global_frame = None


# ==============================
# FLASK LIVE STREAM SERVER
# ==============================

app = Flask(__name__)


def generate_frames():

    global global_frame

    while True:

        if global_frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', global_frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def start_server():
    app.run(host='0.0.0.0', port=5000)


# Start Flask server in background
threading.Thread(target=start_server).start()


# ==============================
# EMOTION DETECTION LOOP
# ==============================

sad_start = None

print("=================================")
print("AI Emotion Monitoring System")
print("Single Camera Mode")
print("Live Stream + Detection Running")
print("Press Q to exit")
print("=================================")


while True:

    ret, frame = camera.read()

    if not ret:
        break

    global_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (64,64))
        face = face / 255.0
        face = np.reshape(face, (1,64,64,1))

        prediction = emotion_model.predict(face)

        emotion = emotion_labels[np.argmax(prediction)]

        print("Detected Emotion:", emotion)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        # SAD TIMER

        if emotion == "Sad":

            if sad_start is None:
                sad_start = time.time()

            if time.time() - sad_start >= 2:

                print("Sad detected for 2 seconds")

                cv2.imwrite("alert.jpg", frame)

                send_snapshot()

                send_live_link()

                send_live_location()

                sad_start = None

        else:
            sad_start = None


    cv2.imshow("Emotion Detection Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
