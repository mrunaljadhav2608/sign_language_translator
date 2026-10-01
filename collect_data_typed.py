"""
collect_data_typed.py
Same as collect_data.py, but lets you type the gesture label instead of
pressing a number key -- use this once your vocabulary grows past 10 words
(vocabulary.py).

Controls:
  - Press SPACE to stop the camera and type the label for the gesture you're
    about to record in the console, then it starts recording immediately.
  - Press 's' while recording to stop and go idle.
  - Press 'q' to quit.
"""

import cv2
import csv
import os
from hand_tracker import HandTracker
from vocabulary import GESTURES

DATA_PATH = "data/gesture_dataset_v2.csv"


def main():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(DATA_PATH)

    tracker = HandTracker(num_hands=2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("Known vocabulary:", GESTURES)
    print("Press SPACE to type a new label and start recording, 's' to stop, 'q' to quit.")

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(126)] + ["label"]
            writer.writerow(header)

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

            label_text = f"Recording: {current_label}" if current_label else "Press SPACE to label"
            cv2.putText(annotated, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imshow("Collect Data (Typed Labels)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_label = None
            elif key == ord(' '):
                current_label = input("Type the word label for this gesture: ").strip().lower()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print(f"Saved dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
