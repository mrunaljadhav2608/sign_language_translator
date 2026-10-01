"""
text_to_speech.py
Thin wrapper around pyttsx3 for offline text-to-speech (uses Windows SAPI5
voices on Windows, so no internet connection or API key is needed).
"""

import pyttsx3


class TextToSpeech:
    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
