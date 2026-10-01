"""
hand_tracker.py  (v2 — two-hand support)
Shared MediaPipe Hand Landmarker wrapper used across the project.

This version tracks up to 2 hands and provides extract_features_multi(),
which combines both hands into a single 126-length feature vector
(63 per hand). If only one hand is visible, the second half is zero-padded,
so existing single-hand gestures still work without retraining -- but for
true two-handed signs you'll want to recollect data with both hands visible
and retrain (see collect_data.py / train_model.py).

Download the model file once from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
and place it in the project root as hand_landmarker.task
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, num_hands=2, min_detection_confidence=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.num_hands = num_hands

    def process_frame(self, frame_bgr):
        """
        Returns (hands_landmarks, annotated_frame).
        hands_landmarks: list of hands, sorted left-to-right by x position for
        consistency, each hand a list of 21 (x, y, z) tuples.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        hands_landmarks = []
        annotated = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        if result.hand_landmarks:
            raw_hands = [[(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_landmarks]
            # sort by average x so "left hand in frame" is always index 0 -- keeps
            # feature ordering consistent between training and inference
            raw_hands.sort(key=lambda pts: sum(p[0] for p in pts) / len(pts))
            hands_landmarks = raw_hands

            for hand in hands_landmarks:
                for lm in hand:
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

        return hands_landmarks, annotated

    def extract_features(self, landmarks):
        """Single-hand 63-length normalized feature vector (unchanged, for
        backward compatibility with existing single-hand models)."""
        if not landmarks:
            return None
        arr = np.array(landmarks)
        wrist = arr[0]
        arr = arr - wrist
        max_val = np.max(np.abs(arr)) or 1.0
        arr = arr / max_val
        return arr.flatten()

    def extract_features_multi(self, hands_landmarks):
        """
        Combines up to 2 hands into a single 126-length feature vector
        (63 per hand, zero-padded if a hand is missing). Use this for
        two-handed gesture training/inference.
        """
        vec = np.zeros(126)
        for i, hand in enumerate(hands_landmarks[:2]):
            features = self.extract_features(hand)
            if features is not None:
                vec[i * 63:(i + 1) * 63] = features
        return vec

    def close(self):
        self.landmarker.close()
