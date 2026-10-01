"""
web_app.py  (v2 — history log + confidence meter)
Flask web UI tying Sign-to-Speech and Speech-to-Sign into one app, now with
a live confidence meter and a running session history log you can download.

Run with:
    python web_app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import threading
import time
from datetime import datetime

import cv2
import joblib
import numpy as np
import speech_recognition as sr
from flask import Flask, Response, jsonify, render_template, request, send_file

from hand_tracker import HandTracker
from text_to_speech import TextToSpeech

MODEL_PATH = "models/gesture_classifier.pkl"
SEQUENCES_PATH = "data/sign_sequences.json"
CONFIDENCE_THRESHOLD = 0.7
STABLE_FRAMES_REQUIRED = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "mode": "sign",
    "recognized_text": "",
    "status": "Idle",
    "listening": False,
    "confidence": 0.0,
    "history": [],   # list of {"time": ..., "direction": ..., "text": ...}
}

latest_jpeg = None
frame_lock = threading.Lock()

clf = joblib.load(MODEL_PATH) if os.path.isfile(MODEL_PATH) else None
sign_library = {}
if os.path.isfile(SEQUENCES_PATH):
    with open(SEQUENCES_PATH, "r") as f:
        sign_library = json.load(f)

tracker = HandTracker()
tts = TextToSpeech()


def log_history(direction, text):
    with state_lock:
        state["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "direction": direction,
            "text": text,
        })
        state["history"] = state["history"][-50:]  # keep last 50 entries


def speak_async(text):
    threading.Thread(target=tts.speak, args=(text,), daemon=True).start()


def draw_skeleton(landmarks, size=(480, 640, 3)):
    canvas = np.zeros(size, dtype=np.uint8)
    h, w, _ = size
    points = [(int(x * w), int(y * h)) for x, y, z in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, points[a], points[b], (100, 220, 130), 3)
    for p in points:
        cv2.circle(canvas, p, 5, (180, 120, 240), -1)
    return canvas


def camera_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("!!! CAMERA FAILED TO OPEN — check if another program/script is using it.")
    else:
        print(">>> Camera opened successfully.")

    last_spoken = None
    stable_label = None
    stable_count = 0

    playback_sequence = None
    playback_idx = 0

    while True:
        with state_lock:
            mode = state["mode"]

        if mode == "sign":
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            hands, annotated = tracker.process_frame(frame)
            predicted_label = None
            confidence = 0.0

            if hands and clf is not None:
                features = tracker.extract_features(hands[0])
                if features is not None:
                    probs = clf.predict_proba([features])[0]
                    best_idx = probs.argmax()
                    confidence = float(probs[best_idx])
                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_label = clf.classes_[best_idx]

            with state_lock:
                state["confidence"] = confidence

            if predicted_label == stable_label:
                stable_count += 1
            else:
                stable_label = predicted_label
                stable_count = 1

            if stable_count == STABLE_FRAMES_REQUIRED and stable_label is not None:
                if stable_label != last_spoken:
                    spoken_text = stable_label.replace("_", " ")
                    speak_async(spoken_text)
                    last_spoken = stable_label
                    with state_lock:
                        state["recognized_text"] = spoken_text
                        state["status"] = "Recognized!"
                    log_history("Sign -> Speech", spoken_text)
            elif stable_label is None:
                last_spoken = None

            label_text = predicted_label if predicted_label else "..."
            cv2.putText(annotated, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (100, 220, 130), 2)
            out_frame = annotated

        else:  # speech mode
            with state_lock:
                seq_to_play = state.pop("play_sequence", None) if "play_sequence" in state else None
            if seq_to_play is not None:
                playback_sequence = seq_to_play
                playback_idx = 0

            if playback_sequence is not None and playback_idx < len(playback_sequence):
                out_frame = draw_skeleton(playback_sequence[playback_idx])
                playback_idx += 1
                if playback_idx >= len(playback_sequence):
                    playback_sequence = None
            else:
                out_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(out_frame, "Press Listen and speak a word", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 120, 240), 2)

        ret2, buf = cv2.imencode(".jpg", out_frame)
        if ret2:
            with frame_lock:
                latest_jpeg = buf.tobytes()

        time.sleep(0.03)


def mjpeg_generator():
    while True:
        with frame_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


def listen_and_play():
    with state_lock:
        state["listening"] = True
        state["status"] = "Listening..."

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
        text = recognizer.recognize_google(audio).lower()
    except sr.WaitTimeoutError:
        text = ""
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""

    with state_lock:
        state["listening"] = False
        state["recognized_text"] = text if text else "(could not understand)"
        state["status"] = "Playing sign..." if text else "Try again"

    if text:
        log_history("Speech -> Sign", text)

    for word in text.split():
        if word in sign_library:
            with state_lock:
                state["play_sequence"] = sign_library[word]
            time.sleep(len(sign_library[word]) * 0.03 + 0.4)


@app.route("/")
def index():
    return render_template("index.html", known_words=sorted(sign_library.keys()))


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_mode", methods=["POST"])
def set_mode():
    mode = request.json.get("mode")
    if mode in ("sign", "speech"):
        with state_lock:
            state["mode"] = mode
            state["recognized_text"] = ""
            state["status"] = "Idle"
    return jsonify({"ok": True, "mode": mode})


@app.route("/listen", methods=["POST"])
def listen():
    with state_lock:
        already = state["listening"]
    if not already:
        threading.Thread(target=listen_and_play, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "mode": state["mode"],
            "recognized_text": state["recognized_text"],
            "status": state["status"],
            "listening": state["listening"],
            "confidence": state["confidence"],
            "history": list(reversed(state["history"][-10:])),
        })


@app.route("/download_history")
def download_history():
    with state_lock:
        lines = [f"[{h['time']}] {h['direction']}: {h['text']}" for h in state["history"]]
    out_path = "session_history.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) if lines else "No history yet.")
    return send_file(out_path, as_attachment=True)


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(debug=False, threaded=True)
