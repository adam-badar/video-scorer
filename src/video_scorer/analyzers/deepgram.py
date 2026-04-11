"""Deepgram Nova-3 transcription analyzer: WPM, fillers, sentiment, topics, language detection."""

import statistics
import time

import httpx

from video_scorer.config import settings

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
RETRY_DELAYS = [0, 5, 15]


async def transcribe(wav_path: str) -> dict | None:
    """Send audio to Deepgram Nova-3 and return parsed results.

    Returns None if API key not set or all retries fail.
    """
    if not settings.deepgram_api_key:
        return None

    params = {
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "true",
        "utterances": "true",
        "filler_words": "true",
        "detect_topics": "true",
        "sentiment": "true",
        "detect_language": "true",
        "mip_opt_out": "true",
    }

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    headers = {
        "Authorization": f"Token {settings.deepgram_api_key.get_secret_value()}",
        "Content-Type": "audio/wav",
    }

    for attempt, delay in enumerate(RETRY_DELAYS):
        if delay > 0:
            time.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    DEEPGRAM_URL,
                    params=params,
                    headers=headers,
                    content=audio_bytes,
                )
            if resp.status_code == 429:
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == len(RETRY_DELAYS) - 1:
                return None
    return None


def analyze_transcript(deepgram_response: dict) -> dict:
    """Extract pacing, filler, sentiment, topic, and language metrics from Deepgram response."""
    result = deepgram_response.get("results", {})
    channels = result.get("channels", [{}])
    if not channels:
        return _empty_analysis()

    channel = channels[0]
    alternatives = channel.get("alternatives", [{}])
    if not alternatives:
        return _empty_analysis()

    alt = alternatives[0]
    words = alt.get("words", [])
    transcript = alt.get("transcript", "")

    # Language detection
    detected_lang = result.get("metadata", {}).get("detected_language")
    if not detected_lang:
        detected_lang = channel.get("detected_language")
    lang_confidence = result.get("metadata", {}).get("language_confidence", 0.0)

    is_english = (
        detected_lang is not None
        and detected_lang.startswith("en")
        and lang_confidence >= 0.7
    )

    # If not English, null out all language-dependent metrics
    if not is_english and words:
        return {
            "transcript": transcript,
            "word_count": len(words),
            "wpm": None,
            "filler_word_count": None,
            "filler_words": None,
            "max_silence_gap_ms": None,
            "pace_variation_stddev": None,
            "detected_language": detected_lang,
            "language_confidence": lang_confidence,
            "language_warning": f"Non-English detected ({detected_lang}, confidence {lang_confidence:.2f}). Language-dependent metrics nulled.",
            "sentiment": None,
            "topics": None,
        }

    # WPM calculation
    if len(words) >= 2:
        first_word_start = words[0]["start"]
        last_word_end = words[-1]["end"]
        speaking_duration = last_word_end - first_word_start
        wpm = round((len(words) / speaking_duration) * 60, 1) if speaking_duration > 0 else 0.0
    else:
        wpm = 0.0

    # Filler words
    filler_list = [w["word"] for w in words if w.get("type") == "filler" or w.get("punctuated_word", "").lower() in ("um", "uh", "mhmm", "uh-huh")]
    filler_count = len(filler_list)

    # Pace variation (WPM per utterance)
    utterances = deepgram_response.get("results", {}).get("utterances", [])
    per_utterance_wpm = []
    for utt in utterances:
        utt_words = utt.get("words", [])
        if len(utt_words) >= 2:
            dur = utt["end"] - utt["start"]
            if dur > 0:
                per_utterance_wpm.append((len(utt_words) / dur) * 60)
    pace_stddev = round(statistics.stdev(per_utterance_wpm), 1) if len(per_utterance_wpm) >= 2 else 0.0

    # Silence gaps between words
    word_gaps_ms = []
    for i in range(1, len(words)):
        gap = (words[i]["start"] - words[i - 1]["end"]) * 1000
        if gap > 0:
            word_gaps_ms.append(int(gap))
    max_word_gap_ms = max(word_gaps_ms, default=0)

    # Sentiment
    sentiment_data = None
    if "sentiments" in result:
        segments = result["sentiments"].get("segments", [])
        if segments:
            sentiments = [s.get("sentiment", "neutral") for s in segments]
            avg_score = sum(s.get("sentiment_score", 0) for s in segments) / len(segments)
            dominant = max(set(sentiments), key=sentiments.count)
            sentiment_data = {"overall": dominant, "confidence": round(avg_score, 2)}

    # Topics
    topics_data = None
    if "topics" in result:
        segments = result["topics"].get("segments", [])
        all_topics = []
        for seg in segments:
            for t in seg.get("topics", []):
                all_topics.append(t.get("topic", ""))
        topics_data = list(set(all_topics))[:10]

    return {
        "transcript": transcript,
        "word_count": len(words),
        "wpm": wpm,
        "filler_word_count": filler_count,
        "filler_words": list(set(filler_list)),
        "max_silence_gap_ms": max_word_gap_ms,
        "pace_variation_stddev": pace_stddev,
        "detected_language": detected_lang,
        "language_confidence": lang_confidence,
        "language_warning": None,
        "sentiment": sentiment_data,
        "topics": topics_data,
    }


def _empty_analysis() -> dict:
    return {
        "transcript": "",
        "word_count": 0,
        "wpm": 0.0,
        "filler_word_count": 0,
        "filler_words": [],
        "max_silence_gap_ms": 0,
        "pace_variation_stddev": 0.0,
        "detected_language": None,
        "language_confidence": 0.0,
        "language_warning": None,
        "sentiment": None,
        "topics": None,
    }
