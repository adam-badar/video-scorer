"""Tests for Deepgram transcript analysis (mocked — no API calls)."""

from video_scorer.analyzers.deepgram import analyze_transcript


def _mock_deepgram_response(words, language="en", lang_confidence=0.95):
    return {
        "results": {
            "metadata": {
                "detected_language": language,
                "language_confidence": lang_confidence,
            },
            "channels": [{
                "alternatives": [{
                    "transcript": " ".join(w["word"] for w in words),
                    "words": words,
                }],
            }],
            "utterances": [{
                "start": words[0]["start"] if words else 0,
                "end": words[-1]["end"] if words else 0,
                "words": words,
            }] if words else [],
        },
    }


def test_wpm_calculation():
    words = [
        {"word": "This", "start": 0.0, "end": 0.3, "type": "word"},
        {"word": "is", "start": 0.4, "end": 0.5, "type": "word"},
        {"word": "a", "start": 0.6, "end": 0.7, "type": "word"},
        {"word": "test", "start": 0.8, "end": 1.0, "type": "word"},
        {"word": "of", "start": 1.1, "end": 1.2, "type": "word"},
        {"word": "the", "start": 1.3, "end": 1.4, "type": "word"},
        {"word": "pipeline", "start": 1.5, "end": 2.0, "type": "word"},
    ]
    result = analyze_transcript(_mock_deepgram_response(words))
    assert result["wpm"] is not None
    assert result["wpm"] > 100  # 7 words in 2 seconds = 210 WPM
    assert result["language_warning"] is None


def test_filler_detection():
    words = [
        {"word": "So", "start": 0.0, "end": 0.2, "type": "word"},
        {"word": "um", "start": 0.3, "end": 0.5, "type": "filler", "punctuated_word": "um"},
        {"word": "I", "start": 0.6, "end": 0.7, "type": "word"},
        {"word": "built", "start": 0.8, "end": 1.0, "type": "word"},
    ]
    result = analyze_transcript(_mock_deepgram_response(words))
    assert result["filler_word_count"] >= 1
    assert "um" in result["filler_words"]


def test_non_english_nulls_all_metrics():
    words = [
        {"word": "Bonjour", "start": 0.0, "end": 0.5, "type": "word"},
        {"word": "tout", "start": 0.6, "end": 0.8, "type": "word"},
        {"word": "le", "start": 0.9, "end": 1.0, "type": "word"},
        {"word": "monde", "start": 1.1, "end": 1.4, "type": "word"},
    ]
    result = analyze_transcript(_mock_deepgram_response(words, language="fr", lang_confidence=0.9))
    assert result["wpm"] is None
    assert result["filler_word_count"] is None
    assert result["sentiment"] is None
    assert result["topics"] is None
    assert result["language_warning"] is not None
    assert "Non-English" in result["language_warning"]


def test_low_confidence_english_nulls_metrics():
    words = [{"word": "test", "start": 0.0, "end": 0.5, "type": "word"}]
    result = analyze_transcript(_mock_deepgram_response(words, language="en", lang_confidence=0.5))
    assert result["wpm"] is None
    assert result["language_warning"] is not None


def test_empty_transcript():
    result = analyze_transcript({"results": {"channels": [{"alternatives": [{"transcript": "", "words": []}]}]}})
    assert result["wpm"] == 0.0
    assert result["filler_word_count"] == 0
