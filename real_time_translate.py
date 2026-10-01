"""
real_time_translate.py
------------------------
Step 3: Real-time gesture-to-text (and optional speech) translation.

Loads the trained model, reads live webcam frames, predicts the gesture
being shown, applies temporal smoothing to avoid flicker, and builds up
a sentence. Reserve two gestures in your training data as special
commands: "space" and "delete" (or just customize SPECIAL_GESTURES below).

Usage:
    python real_time_translate.py

Controls:
    - Hold a gesture steadily; it appears once confidence + smoothing agree
    - Press 'c' to clear the sentence
    - Press 'v' to speak the current sentence aloud (requires pyttsx3)
    - Press 'q' to quit
"""

import cv2
import joblib
import numpy as np
from collections import deque, Counter
from hand_tracker import create_detector, frame_to_mp_image, draw_landmarks, landmarks_to_flat_list

MODEL_PATH = "model/gesture_model.pkl"
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 8       # number of frames to agree on before accepting a prediction
COOLDOWN_FRAMES = 20       # frames to wait before accepting the SAME gesture again

# Optional: gestures you trained that should act as commands instead of letters
SPECIAL_GESTURES = {
    "space": " ",
    "delete": "__DELETE__",
}


def main():
    model = joblib.load(MODEL_PATH)
    detector = create_detector(num_hands=1)

    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_available = True
    except Exception:
        tts_available = False
        print("pyttsx3 not available — speech output disabled.")

    cap = cv2.VideoCapture(0)
    sentence = ""
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    cooldown = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = frame_to_mp_image(rgb)
        result = detector.detect(mp_image)

        current_label = None
        current_conf = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            features = np.array(landmarks_to_flat_list(hand_landmarks)).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            best_idx = np.argmax(probs)
            current_conf = probs[best_idx]
            current_label = model.classes_[best_idx]

            if current_conf >= CONFIDENCE_THRESHOLD:
                recent_predictions.append(current_label)
            else:
                recent_predictions.append(None)
        else:
            recent_predictions.append(None)

        if cooldown > 0:
            cooldown -= 1

        # Only accept a prediction if the smoothing window mostly agrees
        if len(recent_predictions) == SMOOTHING_WINDOW and cooldown == 0:
            counts = Counter(recent_predictions)
            most_common, freq = counts.most_common(1)[0]
            if most_common is not None and freq >= int(SMOOTHING_WINDOW * 0.7):
                if most_common in SPECIAL_GESTURES:
                    action = SPECIAL_GESTURES[most_common]
                    if action == "__DELETE__":
                        sentence = sentence[:-1]
                    else:
                        sentence += action
                else:
                    sentence += most_common
                cooldown = COOLDOWN_FRAMES
                recent_predictions.clear()

        # UI overlay
        display_label = f"{current_label} ({current_conf:.2f})" if current_label else "..."
        cv2.putText(frame, f"Detected: {display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, sentence, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "c=clear  v=speak  q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Language Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            sentence = ""
        elif key == ord('v') and tts_available:
            tts_engine.say(sentence)
            tts_engine.runAndWait()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
