"""
record_sign_sequences.py
Setup tool for Direction B. Records a short landmark sequence for each
word/sign so it can be replayed later as a skeletal animation by
speech_to_sign.py.

Controls:
  - Press 'r' to start recording a sequence, sign the word, press 'r' again
    to stop; you will then be prompted in the console for the word label.
  - Press 'q' to quit.
"""

import cv2
import json
import os
from hand_tracker import HandTracker

OUT_PATH = "data/sign_sequences.json"


def load_library():
    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            return json.load(f)
    return {}


def save_library(library):
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(library, f)


def main():
    library = load_library()
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    recording = False
    current_sequence = []

    print(f"Loaded {len(library)} existing signs: {list(library.keys())}")
    print("Press 'r' to start/stop recording, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hands, annotated = tracker.process_frame(frame)

        if recording and hands:
            current_sequence.append(hands[0])  # one frame's 21 (x, y, z) landmarks

        status = "RECORDING" if recording else "Idle"
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Record Sign Sequences", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording:
                recording = True
                current_sequence = []
            else:
                recording = False
                word = input("Enter the word label for this sequence: ").strip().lower()
                if word:
                    library[word] = current_sequence
                    save_library(library)
                    print(f"Saved sequence for '{word}' ({len(current_sequence)} frames)")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
