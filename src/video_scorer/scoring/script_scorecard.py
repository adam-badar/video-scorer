"""Script scorecard — pre-filming text analysis with weighted scoring."""

import re

from video_scorer.scoring.hooks import GREETING_WORDS, CONTRAST_MARKERS
from video_scorer.scoring.scorecard import PLATFORM_TARGETS, DEFAULT_TARGETS, _grade_from_pct

# CTA keywords — detected via word-boundary regex to avoid false positives
# (e.g., "country" should not match "try", "LinkedIn" should not match "link")
# Multi-word phrases listed first for priority matching
CTA_KEYWORDS = ["check it out", "check out", "sign up", "link in bio", "subscribe", "follow", "comment", "download", "try it", "link below"]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences by common delimiters."""
    sentences = re.split(r'[.!?\n]+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _estimate_duration(word_count: int, platform: str) -> tuple[float, float]:
    """Estimate duration at platform target WPM midpoint.

    Returns (estimated_duration_seconds, target_wpm_midpoint).
    """
    targets = PLATFORM_TARGETS.get(platform, DEFAULT_TARGETS)
    wpm_low, wpm_high = targets["wpm_range"]
    target_wpm = (wpm_low + wpm_high) / 2
    duration = (word_count / target_wpm) * 60
    return round(duration, 1), target_wpm


def compute_script_scorecard(script_text: str, platform: str, gemini_result: dict | None = None) -> dict:
    """Score a script before filming.

    Categories (100 pts total):
    - Hook (30 pts): first sentence length, no greeting, contrast/curiosity marker
    - Structure (30 pts): sections detected (Gemini), CTA placement
    - Pacing (20 pts): estimated duration in platform range
    - Focus (20 pts): narrative thread (Gemini)

    Video-only metrics are NOT scored (loudness, cuts, loop, silence, true peak).
    """
    targets = PLATFORM_TARGETS.get(platform, DEFAULT_TARGETS)
    words = _word_count(script_text)
    estimated_duration, target_wpm = _estimate_duration(words, platform)
    sentences = _split_sentences(script_text)
    script_lower = script_text.lower()

    metrics = []

    # --- Hook (30 pts) ---
    # First sentence ≤ 8 words (punchy hook)
    first_sentence_short = None
    if sentences:
        first_sentence_short = _word_count(sentences[0]) <= 8
    metrics.append(("hook", "first_sentence_punchy", first_sentence_short, 10))

    # No greeting words
    no_greeting = None
    if sentences:
        first_words = sentences[0].lower().split()
        if first_words:
            first_word = first_words[0].strip(".,!?")
            first_two = " ".join(w.strip(".,!?") for w in first_words[:2])
            no_greeting = first_word not in GREETING_WORDS and first_two not in GREETING_WORDS
    metrics.append(("hook", "no_greeting", no_greeting, 10))

    # Contrast/curiosity marker in first 2 sentences
    has_contrast = None
    if len(sentences) >= 1:
        first_two_sentences = " ".join(sentences[:2]).lower()
        first_two_words = set(first_two_sentences.split())
        # Strip punctuation from words for matching
        first_two_clean = {w.strip(".,!?") for w in first_two_words}
        has_contrast = bool(first_two_clean & CONTRAST_MARKERS)
    metrics.append(("hook", "contrast_marker", has_contrast, 10))

    # --- Structure (30 pts) ---
    # Sections detected via Gemini (15 pts) — null if Gemini unavailable
    sections_detected = None
    if gemini_result is not None and "sections_detected" in gemini_result:
        detected = gemini_result["sections_detected"]
        # Full credit if 3+ of [problem, solution, proof, cta] detected
        # Empty list from Gemini means no sections found — score 0, not null
        expected = {"problem", "solution", "proof", "cta"}
        match_count = len(set(detected) & expected) if isinstance(detected, list) else 0
        sections_detected = match_count >= 3
    metrics.append(("structure", "sections_detected", sections_detected, 15))

    # CTA in last 1-2 sentences, not in first 1-2 sentences
    # Use word-boundary regex to avoid false positives (e.g., "country" matching "try")
    sentences_lower = [s.lower() for s in sentences]

    def _has_cta(text: str) -> bool:
        for kw in CTA_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return True
        return False

    last_sentences = " ".join(sentences_lower[-2:]) if len(sentences_lower) >= 2 else " ".join(sentences_lower)
    has_cta_end = _has_cta(last_sentences)
    metrics.append(("structure", "cta_positioned", has_cta_end, 10))

    first_sentences = " ".join(sentences_lower[:2]) if len(sentences_lower) >= 2 else " ".join(sentences_lower)
    no_cta_start = not _has_cta(first_sentences)
    metrics.append(("structure", "no_early_cta", no_cta_start, 5))

    # --- Pacing (20 pts) ---
    # Estimated duration in platform range
    dur_low, dur_high = targets["duration_range"]
    duration_in_range = dur_low <= estimated_duration <= dur_high
    metrics.append(("pacing", "duration_in_range", duration_in_range, 10))

    # Script length reasonable (not too short or too long)
    length_ok = words >= 30 and estimated_duration <= dur_high * 1.5
    metrics.append(("pacing", "length_ok", length_ok, 10))

    # --- Focus (20 pts) — Gemini-dependent ---
    focus_single = None
    focus_no_tangents = None
    if gemini_result:
        focus_val = gemini_result.get("focus_score", "")
        if focus_val == "single_thread":
            focus_single = True
            focus_no_tangents = True
        elif focus_val == "branches":
            focus_single = False
            focus_no_tangents = True
        elif focus_val == "unfocused":
            focus_single = False
            focus_no_tangents = False
    metrics.append(("focus", "single_thread", focus_single, 15))
    metrics.append(("focus", "no_tangents", focus_no_tangents, 5))

    # --- Aggregate ---
    total_scored = 0
    total_max = 0
    category_scores: dict[str, dict] = {}

    for cat, name, value, pts in metrics:
        if cat not in category_scores:
            category_scores[cat] = {"scored": 0, "max": 0, "checks": []}
        if value is not None:
            scored = pts if value else 0
            total_scored += scored
            total_max += pts
            category_scores[cat]["scored"] += scored
            category_scores[cat]["max"] += pts
            category_scores[cat]["checks"].append({"name": name, "passed": value, "points": pts, "scored": scored})
        else:
            category_scores[cat]["checks"].append({"name": name, "passed": None, "points": pts, "scored": None})

    percentage = (total_scored / total_max * 100) if total_max > 0 else 0
    grade = _grade_from_pct(percentage)

    return {
        "total": total_scored,
        "max_possible": total_max,
        "grade": grade,
        "percentage": round(percentage, 1),
        "word_count": words,
        "estimated_duration_seconds": estimated_duration,
        "target_wpm": target_wpm,
        "platform": platform,
        "breakdown": {cat: data["scored"] for cat, data in category_scores.items()},
        "categories": category_scores,
        "platform_targets": {
            "platform": platform,
            "wpm_target": f"{targets['wpm_range'][0]}-{targets['wpm_range'][1]}",
            "duration_target": f"{targets['duration_range'][0]}-{targets['duration_range'][1]}s",
            "duration_in_range": duration_in_range,
        },
    }
