"""
sign_to_speech_lstm.py
Real-time Sign-to-Speech using the trained LSTM motion model instead of the
static-pose RandomForest. Keeps a rolling buffer of the last 30 frames and
feeds the whole window into the LSTM every frame, which lets it recognize
gestures that involve movement, not just a held pose.

Run this after train_lstm_model.py has produced models/gesture_lstm.keras.
"""

import collections
import cv2
import numpy as np
import tensorflow as tf
from hand_tracker import HandTracker
from text_to_speech import TextToSpeech

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126
MODEL_PATH = "models/gesture_lstm.keras"
LABELS_PATH = "models/lstm_labels.txt"
CONFIDENCE_THRESHOLD = 0.75
STABLE_PREDICTIONS_REQUIRED = 5


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f if line.strip()]

    tracker = HandTracker(num_hands=2)
    tts = TextToSpeech()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    last_spoken = None
    stable_label = None
    stable_count = 0

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hands, annotated = tracker.process_frame(frame)

        features = tracker.extract_features_multi(hands) if hands else np.zeros(FEATURE_DIM)
        buffer.append(features)

        predicted_label = None
        if len(buffer) == SEQUENCE_LENGTH:
            window = np.expand_dims(np.array(buffer), axis=0)  # (1, 30, 126)
            probs = model.predict(window, verbose=0)[0]
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = labels[best_idx]

        if predicted_label == stable_label:
            stable_count += 1
        else:
            stable_label = predicted_label
            stable_count = 1

        if stable_count == STABLE_PREDICTIONS_REQUIRED and stable_label is not None:
            if stable_label != last_spoken:
                print("Recognized:", stable_label)
                tts.speak(stable_label.replace("_", " "))
                last_spoken = stable_label
                buffer.clear()
        elif stable_label is None:
            last_spoken = None

        display_text = predicted_label if predicted_label else "..."
        cv2.putText(annotated, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow("Sign to Speech (LSTM)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
