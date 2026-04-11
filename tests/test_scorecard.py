"""Tests for scorecard aggregation and grade renormalization."""

from video_scorer.scoring.scorecard import compute_scorecard


def _full_pacing():
    return {"wpm": 185.0, "filler_word_count": 0}


def _full_editing():
    return {"cuts_per_minute": 10.0, "cut_timestamps": [3.0, 7.0, 12.0], "total_cuts": 3}


def _full_audio():
    return {"loudness_lufs": -14.2, "true_peak_dbtp": -1.8, "silence_ratio": 0.03, "max_silence_gap_ms": 200}


def _full_hooks():
    return {"has_content_word_in_first_2s": True, "no_greeting_preamble": True, "has_contrast_marker_in_first_3s": True}


def test_full_scorecard_grade_a():
    score = compute_scorecard(
        pacing=_full_pacing(),
        editing=_full_editing(),
        audio=_full_audio(),
        hook_checks=_full_hooks(),
        loop_score=0.85,
        duration_seconds=45.0,
        platform="tiktok",
    )
    assert score["grade"] == "A"
    assert score["max_possible"] == 130
    assert score["total"] > 0


def test_renormalization_deepgram_unavailable():
    """When Deepgram fails, all Deepgram-dependent metrics are None. Grade uses only local metrics."""
    score = compute_scorecard(
        pacing={"wpm": None, "filler_word_count": None},
        editing=_full_editing(),
        audio=_full_audio(),
        hook_checks={"has_content_word_in_first_2s": None, "no_greeting_preamble": None, "has_contrast_marker_in_first_3s": None},
        loop_score=0.85,
        duration_seconds=45.0,
        platform="tiktok",
    )
    # Hook (40) + Pacing WPM (15) + Pacing fillers (10) = 65 excluded
    assert score["max_possible"] == 130 - 65  # = 65
    assert score["grade"] in {"A", "B", "C", "D", "F"}


def test_zero_vs_null():
    """WPM = 0 (silent video, measured) counts against grade. WPM = None (unavailable) is excluded."""
    # Zero WPM — measured, counts as 0
    score_zero = compute_scorecard(
        pacing={"wpm": 0.0, "filler_word_count": 0},
        editing=_full_editing(),
        audio=_full_audio(),
        hook_checks={"has_content_word_in_first_2s": False, "no_greeting_preamble": True, "has_contrast_marker_in_first_3s": False},
        loop_score=0.85,
        duration_seconds=45.0,
        platform="tiktok",
    )

    # Null WPM — unavailable, excluded
    score_null = compute_scorecard(
        pacing={"wpm": None, "filler_word_count": None},
        editing=_full_editing(),
        audio=_full_audio(),
        hook_checks={"has_content_word_in_first_2s": None, "no_greeting_preamble": None, "has_contrast_marker_in_first_3s": None},
        loop_score=0.85,
        duration_seconds=45.0,
        platform="tiktok",
    )

    # Zero has full denominator, null has reduced denominator
    assert score_zero["max_possible"] == 130
    assert score_null["max_possible"] < 130


def test_platform_targets_differ():
    """TikTok and LinkedIn have different WPM targets."""
    score_tiktok = compute_scorecard(
        pacing={"wpm": 190.0, "filler_word_count": 0},
        editing={"cuts_per_minute": 10.0},
        audio=_full_audio(),
        hook_checks=_full_hooks(),
        loop_score=0.8,
        duration_seconds=45.0,
        platform="tiktok",
    )
    score_linkedin = compute_scorecard(
        pacing={"wpm": 190.0, "filler_word_count": 0},
        editing={"cuts_per_minute": 10.0},
        audio=_full_audio(),
        hook_checks=_full_hooks(),
        loop_score=0.8,
        duration_seconds=45.0,
        platform="linkedin",
    )
    # 190 WPM is in range for TikTok (180-220) but out of range for LinkedIn (140-170)
    assert score_tiktok["platform_targets"]["wpm_in_range"] is True
    assert score_linkedin["platform_targets"]["wpm_in_range"] is False
