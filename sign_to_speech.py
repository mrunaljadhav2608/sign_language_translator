"""
sign_to_speech.py
Direction A: webcam gestures -> recognized text -> spoken aloud.

Press 'q' to quit.
"""

import cv2
import joblib
from hand_tracker import HandTracker
from text_to_speech import TextToSpeech

MODEL_PATH = "models/gesture_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.7
STABLE_FRAMES_REQUIRED = 8  # gesture must hold steady this many frames before it is spoken


def main():
    clf = joblib.load(MODEL_PATH)
    tracker = HandTracker()
    tts = TextToSpeech()

    cap = cv2.VideoCapture(0)

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
        predicted_label = None

        if hands:
            features = tracker.extract_features(hands[0])
            if features is not None:
                probs = clf.predict_proba([features])[0]
                best_idx = probs.argmax()
                confidence = probs[best_idx]
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = clf.classes_[best_idx]

        # Debounce so the same gesture held for many frames only speaks once,
        # and briefly-flickered mispredictions don't get spoken at all.
        if predicted_label == stable_label:
            stable_count += 1
        else:
            stable_label = predicted_label
            stable_count = 1

        if stable_count == STABLE_FRAMES_REQUIRED and stable_label is not None:
            if stable_label != last_spoken:
                print("Recognized:", stable_label)
                tts.speak(stable_label.replace("_", " "))
                last_spoken = stable_label
        elif stable_label is None:
            last_spoken = None  # allow the same word to be spoken again after a pause

        display_text = predicted_label if predicted_label else "..."
        cv2.putText(annotated, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow("Sign to Speech", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
