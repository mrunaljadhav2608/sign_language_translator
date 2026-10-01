"""
speech_to_sign.py
Direction B: microphone speech -> text -> replay recorded sign landmark
sequences (from record_sign_sequences.py) as a skeletal animation.

Say "quit" to exit.
"""

import json
import time
import os
import cv2
import numpy as np
import speech_recognition as sr

LIBRARY_PATH = "data/sign_sequences.json"
CANVAS_SIZE = (480, 640, 3)

# Standard MediaPipe hand landmark connections (21 points).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def load_library():
    if not os.path.isfile(LIBRARY_PATH):
        print(f"No sign library found at {LIBRARY_PATH}. Run record_sign_sequences.py first.")
        return {}
    with open(LIBRARY_PATH, "r") as f:
        return json.load(f)


def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... speak a sentence.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Heard:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("Speech recognition service error:", e)
        return ""


def draw_skeleton_frame(landmarks):
    canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
    h, w, _ = CANVAS_SIZE
    points = [(int(x * w), int(y * h)) for x, y, z in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(canvas, p, 4, (0, 0, 255), -1)

    return canvas


def play_sequence(sequence, delay=0.03):
    for frame_landmarks in sequence:
        canvas = draw_skeleton_frame(frame_landmarks)
        cv2.imshow("Speech to Sign", canvas)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break


def main():
    library = load_library()
    if not library:
        return
    print(f"Loaded {len(library)} known signs: {list(library.keys())}")

    while True:
        text = listen_and_transcribe()
        if not text:
            continue
        if text.strip() == "quit":
            break

        for word in text.split():
            if word in library:
                print("Signing:", word)
                play_sequence(library[word])
            else:
                print(f"No sign recorded for '{word}', skipping.")
            time.sleep(0.3)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
