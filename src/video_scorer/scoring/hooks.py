"""Machine-checkable hook template verification against transcript timing."""

GREETING_WORDS = {"hey", "hi", "hello", "what's up", "yo", "welcome", "good morning", "good evening"}

CONTRAST_MARKERS = {"stop", "instead", "don't", "never", "but", "however", "forget", "quit", "unlike"}


def check_hooks(words: list[dict], duration_seconds: float) -> dict:
    """Verify hook quality from Deepgram word-level timestamps.

    Args:
        words: List of Deepgram word objects with 'word', 'start', 'end' keys.
        duration_seconds: Total video duration in seconds.

    Returns:
        Dict with hook check results and individual scores.
    """
    if duration_seconds < 3:
        return {
            "has_content_word_in_first_2s": None,
            "no_greeting_preamble": None,
            "has_contrast_marker_in_first_3s": None,
            "duration_warning": "Video shorter than 3s — hook checks skipped",
        }

    if not words:
        return {
            "has_content_word_in_first_2s": False,
            "no_greeting_preamble": True,
            "has_contrast_marker_in_first_3s": False,
        }

    # Check 1: Content word in first 2 seconds
    first_2s_words = [w for w in words if w.get("start", 999) < 2.0]
    has_content = len(first_2s_words) >= 2  # At least 2 words spoken by 2s mark

    # Check 2: No greeting preamble
    first_word = words[0].get("word", "").lower().strip(".,!?")
    first_two = " ".join(w.get("word", "").lower().strip(".,!?") for w in words[:2])
    no_greeting = first_word not in GREETING_WORDS and first_two not in GREETING_WORDS

    # Check 3: Contrast marker in first 3 seconds
    first_3s_words = [w.get("word", "").lower().strip(".,!?") for w in words if w.get("start", 999) < 3.0]
    has_contrast = any(marker in first_3s_words for marker in CONTRAST_MARKERS)

    return {
        "has_content_word_in_first_2s": has_content,
        "no_greeting_preamble": no_greeting,
        "has_contrast_marker_in_first_3s": has_contrast,
    }
