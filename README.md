# ✧ Sign Language ↔ Speech Two-Way Translator ✧

A real-time system that translates in both directions between hand signs
and spoken language — built with MediaPipe, OpenCV, scikit-learn, and
(optionally) an LSTM sequence model, wrapped in a Flask web interface.

> 📸 *Add a demo GIF or screenshot here before publishing to GitHub —
> a 5-10 second clip of the web UI recognizing a sign and speaking it
> out loud makes a huge difference on a resume/portfolio page.*

## Features

- **Sign → Speech**: webcam captures a hand gesture, a trained classifier
  recognizes it, and the result is spoken aloud.
- **Speech → Sign**: microphone captures spoken words, transcribes them,
  and replays a recorded skeletal animation for each recognized word.
- **Web UI**: single-page interface with a live video feed, a mode toggle,
  a live confidence meter, and a downloadable session history log.
- **Two-hand support**: feature pipeline supports up to two hands per
  gesture (`hand_tracker.py` v2).
- **LSTM upgrade path**: optional motion-aware sequence model for gestures
  that involve movement, not just a held pose.

## Project structure

```
sign_language_translator/
├── hand_tracker.py            Shared MediaPipe wrapper (supports 2 hands)
├── vocabulary.py              Central list of trained words/gestures
├── collect_data.py            Record static-pose gesture samples (0-9 keys)
├── collect_data_typed.py      Same, with typed labels for >10 words
├── collect_sequence_data.py   Record 30-frame motion sequences (for LSTM)
├── train_model.py             Train the RandomForest static-pose classifier
├── train_lstm_model.py        Train the LSTM motion classifier (optional)
├── sign_to_speech.py          Direction A (RandomForest, static poses)
├── sign_to_speech_lstm.py     Direction A (LSTM, motion-aware)
├── record_sign_sequences.py   Record landmark sequences for Speech → Sign
├── speech_to_sign.py          Direction B: speech → sign animation
├── text_to_speech.py          Offline TTS wrapper (pyttsx3)
├── web_app.py                 Flask web UI tying both directions together
├── templates/index.html       Web UI page
├── static/style.css           Web UI styling
├── requirements.txt
├── data/                      Datasets (CSV, JSON, sequence .npy files)
└── models/                    Trained classifiers
```

## Setup

1. **Virtual environment:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   python -m pip install -r requirements.txt
   ```
   For the LSTM upgrade, also run:
   ```
   python -m pip install tensorflow
   ```

3. **Hand landmark model** — download and place in the project root as
   `hand_landmarker.task`:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

## Usage

### Quick path (web app, recommended)
```
python web_app.py
```
Open `http://127.0.0.1:5000`. Toggle between directions, use "Listen" for
Speech → Sign, and download your session history from the panel at the
bottom.

### Static-pose pipeline (RandomForest)
```
python collect_data.py       # record gesture samples
python train_model.py        # train + evaluate the classifier
python sign_to_speech.py     # run live recognition (standalone script)
```

### Motion pipeline (LSTM, optional upgrade)
```
python collect_sequence_data.py   # record 30-frame motion sequences
python train_lstm_model.py        # train the LSTM
python sign_to_speech_lstm.py     # run live recognition with motion awareness
```

### Speech → Sign setup
```
python record_sign_sequences.py   # record a landmark sequence per word
python speech_to_sign.py          # standalone speech-to-sign script
```

## Expanding the vocabulary

Add new words to `vocabulary.py`, then run `collect_data.py` (or
`collect_data_typed.py` if you're past 10 words) to record samples only
for the new entries — no need to redo existing gestures. Re-run
`train_model.py` afterward to retrain on the combined dataset.

## Notes on two-hand support

`hand_tracker.py` now tracks up to 2 hands and combines them into a single
126-length feature vector. Existing single-hand models trained on the old
63-length format are **not** compatible with this — if you want true
two-handed gestures, start a fresh dataset with `collect_data.py` (which
now writes to `data/gesture_dataset_v2.csv`) and retrain.

## Scope tip

Don't try to cover a full vocabulary right away. A focused, well-demoed
15-30 word vocabulary is more convincing in an interview or resume video
than a large but unreliable one.

## Possible next steps

- Swap the skeletal-line animation for a fully rigged 3D avatar.
- Add sentence-level grammar handling instead of word-by-word playback.
- Deploy the Flask app for a live shareable demo link.
- Add automated tests for the landmark normalization/feature extraction logic.

## Requirements addition

Add this line to `requirements.txt` if not already present:
```
tensorflow
```
(only needed if you're using the LSTM upgrade path)

