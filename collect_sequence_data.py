"""
collect_sequence_data.py
Records fixed-length motion sequences (30 frames each) per gesture, for
training the LSTM model in train_lstm_model.py. Unlike collect_data.py
(single static frame per sample), this captures the *motion* of a sign,
which matters for gestures that involve movement rather than a held pose.

Controls:
  - Press a number key (0-9) to select which gesture to record.
  - Each press records ONE 30-frame sequence (about 1 second at 30fps),
    shown with a countdown, then automatically saves it.
  - Repeat the same number key multiple times to record several sequences
    per gesture (aim for at least 20-30 sequences per gesture).
  - Press 'q' to quit.
"""

import cv2
import numpy as np
import os
from hand_tracker import HandTracker
from vocabulary import GESTURES

SEQUENCE_LENGTH = 30
OUT_DIR = "data/sequences"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for g in GESTURES:
        os.makedirs(os.path.join(OUT_DIR, g), exist_ok=True)

    tracker = HandTracker(num_hands=2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    counts = {g: len(os.listdir(os.path.join(OUT_DIR, g))) for g in GESTURES}

    print("Gestures available:")
    for i, g in enumerate(GESTURES):
        print(f"  [{i}] {g}  (existing sequences: {counts[g]})")
    print("Press a number key to record one 30-frame sequence for that gesture, 'q' to quit.")

    recording = False
    current_label = None
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hands, annotated = tracker.process_frame(frame)

        if recording:
            features = tracker.extract_features_multi(hands)
            buffer.append(features)
            cv2.putText(annotated, f"REC {current_label}  {len(buffer)}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if len(buffer) >= SEQUENCE_LENGTH:
                idx = counts[current_label]
                out_path = os.path.join(OUT_DIR, current_label, f"seq_{idx}.npy")
                np.save(out_path, np.array(buffer))
                counts[current_label] += 1
                print(f"Saved {out_path}")
                recording = False
                buffer = []
        else:
            cv2.putText(annotated, "Press a number key to record a sequence", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.imshow("Collect Sequence Data", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9') and not recording:
            idx = key - ord('0')
            if idx < len(GESTURES):
                current_label = GESTURES[idx]
                recording = True
                buffer = []

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
