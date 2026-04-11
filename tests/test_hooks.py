"""Tests for hook template verification."""

from video_scorer.scoring.hooks import check_hooks


def test_good_hook():
    words = [
        {"word": "This", "start": 0.1, "end": 0.3},
        {"word": "video", "start": 0.4, "end": 0.7},
        {"word": "was", "start": 0.8, "end": 1.0},
        {"word": "posted", "start": 1.1, "end": 1.4},
        {"word": "without", "start": 1.5, "end": 1.8},
        {"word": "me", "start": 1.9, "end": 2.1},
        {"word": "ever", "start": 2.2, "end": 2.4},
        {"word": "opening", "start": 2.5, "end": 2.8},
    ]
    result = check_hooks(words, duration_seconds=45.0)
    assert result["has_content_word_in_first_2s"] is True
    assert result["no_greeting_preamble"] is True


def test_greeting_preamble():
    words = [
        {"word": "Hey", "start": 0.1, "end": 0.3},
        {"word": "everyone", "start": 0.4, "end": 0.8},
        {"word": "today", "start": 1.0, "end": 1.3},
    ]
    result = check_hooks(words, duration_seconds=30.0)
    assert result["no_greeting_preamble"] is False


def test_contrast_marker():
    words = [
        {"word": "Stop", "start": 0.1, "end": 0.4},
        {"word": "using", "start": 0.5, "end": 0.8},
        {"word": "Hootsuite", "start": 0.9, "end": 1.4},
    ]
    result = check_hooks(words, duration_seconds=30.0)
    assert result["has_contrast_marker_in_first_3s"] is True


def test_short_video_skips_hooks():
    words = [{"word": "Hi", "start": 0.1, "end": 0.3}]
    result = check_hooks(words, duration_seconds=2.0)
    assert result["has_content_word_in_first_2s"] is None
    assert "duration_warning" in result


def test_empty_words():
    result = check_hooks([], duration_seconds=30.0)
    assert result["has_content_word_in_first_2s"] is False
    assert result["no_greeting_preamble"] is True
