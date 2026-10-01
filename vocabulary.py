"""
vocabulary.py
Central place to manage your gesture/word vocabulary so you don't have to
edit it separately inside collect_data.py, record_sign_sequences.py, etc.

Add new words here as you expand the project, then re-run collect_data.py
to record samples for the new ones (you don't need to redo old gestures --
just add new number-key entries and record only the new words).
"""

# Keep this list under 10 items if you're using single number-key (0-9)
# labeling in collect_data.py. For a larger vocabulary, switch to typing
# the label instead of pressing a number (see collect_data_typed.py below
# in the same folder for that variant once your list grows past 10).
GESTURES = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "please",
    "help",
    "water",
    "sorry",
    # -- add more words below as you expand --
    # "goodbye",
    # "love",
    # "friend",
    # "eat",
    # "family",
]
