"""
app.py
Simple console menu that launches any part of the two-way translator.
(A full graphical UI with embedded live video is a good stretch goal once
the core pipeline works reliably -- this launcher keeps the demo simple
and dependable in the meantime.)
"""

import subprocess
import sys


def main():
    options = {
        "1": ("Sign -> Speech", "sign_to_speech.py"),
        "2": ("Speech -> Sign", "speech_to_sign.py"),
        "3": ("Collect training data (gestures)", "collect_data.py"),
        "4": ("Train gesture classifier", "train_model.py"),
        "5": ("Record a new sign sequence (for Speech -> Sign)", "record_sign_sequences.py"),
    }

    print("=== Sign Language Two-Way Translator ===")
    for key, (label, _) in options.items():
        print(f"  [{key}] {label}")
    print("  [q] Quit")

    choice = input("Choose an option: ").strip().lower()

    if choice in options:
        _, script = options[choice]
        subprocess.run([sys.executable, script])
    elif choice == "q":
        return
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
