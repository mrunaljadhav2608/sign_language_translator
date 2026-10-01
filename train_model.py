"""
train_model.py
Trains a RandomForest classifier on the gesture dataset collected by
collect_data.py and saves it to models/gesture_classifier.pkl.
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "data/gesture_dataset.csv"
MODEL_OUT = "models/gesture_classifier.pkl"


def main():
    os.makedirs("models", exist_ok=True)

    if not os.path.isfile(DATA_PATH):
        print(f"No dataset found at {DATA_PATH}. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report on held-out test set:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
