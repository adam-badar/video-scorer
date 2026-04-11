"""Scorecard aggregation — weighted scoring with grade renormalization."""

PLATFORM_TARGETS = {
    "tiktok": {"wpm_range": (180, 220), "cuts_range": (8, 12), "duration_range": (30, 60)},
    "youtube": {"wpm_range": (160, 200), "cuts_range": (8, 12), "duration_range": (30, 58)},
    "instagram": {"wpm_range": (170, 210), "cuts_range": (8, 12), "duration_range": (15, 30)},
    "linkedin": {"wpm_range": (140, 170), "cuts_range": (4, 6), "duration_range": (30, 90)},
}

DEFAULT_TARGETS = {"wpm_range": (160, 200), "cuts_range": (8, 12), "duration_range": (30, 60)}


def _in_range(value: float | None, low: float, high: float) -> bool | None:
    if value is None:
        return None
    return low <= value <= high


def _score_metric(value: bool | None, points: int) -> tuple[int | None, int]:
    """Return (scored_points, max_points). scored_points is None if value is None."""
    if value is None:
        return None, points
    return (points if value else 0), points


def compute_scorecard(
    pacing: dict,
    editing: dict,
    audio: dict,
    hook_checks: dict,
    loop_score: float,
    duration_seconds: float,
    platform: str = "tiktok",
) -> dict:
    """Compute weighted scorecard with grade renormalization.

    Metrics sourced from Deepgram become null when Deepgram fails or non-English detected.
    Null metrics are excluded from both numerator and denominator.
    """
    targets = PLATFORM_TARGETS.get(platform, DEFAULT_TARGETS)

    wpm = pacing.get("wpm")
    filler_count = pacing.get("filler_word_count")
    max_silence_ms = audio.get("max_silence_gap_ms", 0)
    cuts_per_min = editing.get("cuts_per_minute")
    loudness = audio.get("loudness_lufs")
    true_peak = audio.get("true_peak_dbtp")
    silence_ratio = audio.get("silence_ratio", 0)

    # Hook checks (Deepgram-dependent — can be None)
    hook_content = hook_checks.get("has_content_word_in_first_2s")
    hook_greeting = hook_checks.get("no_greeting_preamble")
    hook_contrast = hook_checks.get("has_contrast_marker_in_first_3s")

    # Compute each metric's score
    metrics = []

    # Hook category (40 pts total, all Deepgram-dependent)
    metrics.append(("hook", "content_first_2s", *_score_metric(hook_content, 15)))
    metrics.append(("hook", "no_greeting", *_score_metric(hook_greeting, 15)))
    metrics.append(("hook", "contrast_marker", *_score_metric(hook_contrast, 10)))

    # Pacing category (30 pts, mostly Deepgram-dependent)
    wpm_in_range = _in_range(wpm, *targets["wpm_range"])
    metrics.append(("pacing", "wpm_in_range", *_score_metric(wpm_in_range, 15)))

    filler_clean = (filler_count == 0) if filler_count is not None else None
    metrics.append(("pacing", "no_fillers", *_score_metric(filler_clean, 10)))

    silence_ok = max_silence_ms < 500
    metrics.append(("pacing", "silence_gaps_ok", *_score_metric(silence_ok, 5)))

    # Editing category (20 pts, all local)
    cuts_in_range = _in_range(cuts_per_min, *targets["cuts_range"])
    metrics.append(("editing", "cuts_in_range", *_score_metric(cuts_in_range, 15)))

    dur_in_range = _in_range(duration_seconds, *targets["duration_range"])
    metrics.append(("editing", "duration_in_range", *_score_metric(dur_in_range, 5)))

    # Audio category (25 pts, all local)
    loudness_ok = (abs(loudness - (-14)) <= 1) if loudness is not None else None
    metrics.append(("audio", "loudness_ok", *_score_metric(loudness_ok, 15)))

    peak_ok = (true_peak <= -1.5) if true_peak is not None else None
    metrics.append(("audio", "true_peak_ok", *_score_metric(peak_ok, 5)))

    silence_ratio_ok = silence_ratio < 0.10
    metrics.append(("audio", "silence_ratio_ok", *_score_metric(silence_ratio_ok, 5)))

    # Structure category (15 pts, all local)
    loop_ok = loop_score > 0.7
    metrics.append(("structure", "loop_score_ok", *_score_metric(loop_ok, 10)))

    dur_sweet = _in_range(duration_seconds, *targets["duration_range"])
    metrics.append(("structure", "duration_sweet_spot", *_score_metric(dur_sweet, 5)))

    # Aggregate
    total_scored = 0
    total_max = 0
    category_scores = {}

    for cat, name, scored, max_pts in metrics:
        if cat not in category_scores:
            category_scores[cat] = {"scored": 0, "max": 0}
        if scored is not None:
            total_scored += scored
            total_max += max_pts
            category_scores[cat]["scored"] += scored
            category_scores[cat]["max"] += max_pts

    percentage = (total_scored / total_max * 100) if total_max > 0 else 0
    grade = _grade_from_pct(percentage)

    # Platform target summary
    platform_info = {
        "platform": platform,
        "wpm_target": f"{targets['wpm_range'][0]}-{targets['wpm_range'][1]}",
        "wpm_in_range": wpm_in_range,
        "cuts_target": f"{targets['cuts_range'][0]}-{targets['cuts_range'][1]}",
        "cuts_in_range": cuts_in_range,
        "duration_target": f"{targets['duration_range'][0]}-{targets['duration_range'][1]}s",
        "duration_in_range": dur_in_range,
        "loudness_target": "-14 LUFS",
        "loudness_in_range": loudness_ok,
    }

    breakdown = {cat: data["scored"] for cat, data in category_scores.items()}

    return {
        "total": total_scored,
        "max_possible": total_max,
        "grade": grade,
        "percentage": round(percentage, 1),
        "breakdown": breakdown,
        "platform_targets": platform_info,
    }


def _grade_from_pct(pct: float) -> str:
    if pct >= 80:
        return "A"
    if pct >= 65:
        return "B"
    if pct >= 50:
        return "C"
    if pct >= 35:
        return "D"
    return "F"
