"""
collect_data.py  (v2 — two-hand support)
Collects labeled gesture samples using the combined 126-length two-hand
feature vector. Works fine for single-hand gestures too (second hand is
zero-padded automatically).

Controls:
  - Press a number key (0-9) to select which gesture you are about to sign;
    every frame while a label is selected gets written to the CSV.
  - Press SPACE to pause recording.
  - Press 'q' to quit.

NOTE: If you already have data/gesture_dataset.csv from the single-hand
(63-column) version, don't append to it with this script -- the column
count won't match. Start a fresh CSV (e.g. data/gesture_dataset_v2.csv)
for two-handed gestures, or move the old file aside first.
"""

import cv2
import csv
import os
from hand_tracker import HandTracker

DATA_PATH = "data/gesture_dataset_v2.csv"

GESTURES = ["hello", "thank_you", "yes", "no", "please", "help", "water", "sorry"]


def main():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(DATA_PATH)

    tracker = HandTracker(num_hands=2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(126)] + ["label"]
            writer.writerow(header)

        print("Gestures available:")
        for i, g in enumerate(GESTURES):
            print(f"  [{i}] {g}")
        print("Press the number key to start labeling that gesture, SPACE to pause, 'q' to quit.")

        current_label = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            hands, annotated = tracker.process_frame(frame)

            if hands and current_label is not None:
                features = tracker.extract_features_multi(hands)
                writer.writerow(list(features) + [current_label])

            label_text = f"Recording: {current_label}" if current_label else "Press a number key to label"
            cv2.putText(annotated, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"Hands detected: {len(hands)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow("Collect Data (Two-Hand)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                idx = key - ord('0')
                if idx < len(GESTURES):
                    current_label = GESTURES[idx]
            elif key == ord(' '):
                current_label = None

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print(f"Saved dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
