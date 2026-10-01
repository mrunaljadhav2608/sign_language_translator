"""
train_lstm_model.py
Trains an LSTM sequence model on the motion sequences recorded by
collect_sequence_data.py. This is the upgrade path from the static-pose
RandomForest classifier to a model that understands *motion*, not just a
single held pose -- necessary for gestures where the movement itself
carries meaning (e.g. a wave, a "come here" motion).

Requires TensorFlow:
    python -m pip install tensorflow

Run:
    python train_lstm_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from vocabulary import GESTURES

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126  # two-hand feature vector from hand_tracker.py
DATA_DIR = "data/sequences"
MODEL_OUT = "models/gesture_lstm.keras"
LABELS_OUT = "models/lstm_labels.txt"


def load_dataset():
    X, y = [], []
    for label in GESTURES:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(folder, fname))
                if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                    X.append(seq)
                    y.append(label)
    return np.array(X), np.array(y)


def main():
    os.makedirs("models", exist_ok=True)

    X, y = load_dataset()
    if len(X) == 0:
        print("No sequence data found. Run collect_sequence_data.py first.")
        return

    print(f"Loaded {len(X)} sequences across {len(set(y))} gestures.")

    label_list = sorted(set(y))
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    y_idx = np.array([label_to_idx[label] for label in y])
    y_cat = to_categorical(y_idx, num_classes=len(label_list))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_idx
    )

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(32, activation="tanh"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(len(label_list), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=8,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.3f}")

    model.save(MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        f.write("\n".join(label_list))
    print(f"Model saved to {MODEL_OUT}")
    print(f"Label order saved to {LABELS_OUT}")


if __name__ == "__main__":
    main()
