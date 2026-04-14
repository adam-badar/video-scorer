"""Tests for script scorecard — pre-filming text analysis."""

from video_scorer.scoring.script_scorecard import compute_script_scorecard


SAMPLE_SCRIPT = """I built seven products this year. Without a team.

A sales outreach platform. An email infrastructure for AI agents. A personal AI that captures everything I see and hear.

The tools? Claude Code and Cursor. Both cost $20 a month.

If you're still writing code line by line, you're already behind.

Follow for more."""

GREETING_SCRIPT = """Hey everyone, welcome to my channel!

Today I want to talk about how I built seven products this year.

A sales outreach platform. An email infrastructure for AI agents.

Follow for more."""

SHORT_SCRIPT = "Check this out. Subscribe now."

CTA_SHORT_SCRIPT = """This is the main content of my video about building products.

Subscribe now."""


def test_script_scorecard_max_possible_is_100_with_gemini():
    """Script scoring has max 100 points, not 130 (no video-only metrics)."""
    gemini_result = {
        "sections_detected": ["problem", "solution", "proof", "cta"],
        "focus_score": "single_thread",
    }
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", gemini_result)
    assert result["max_possible"] == 100


def test_script_scorecard_max_65_without_gemini():
    """Without Gemini, focus (20 pts) and structure.sections (15 pts) are null — max drops to 65."""
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", gemini_result=None)
    assert result["max_possible"] == 65


def test_hook_detection_no_greeting():
    """Script starting with 'Hey everyone' scores 0 on no_greeting."""
    result = compute_script_scorecard(GREETING_SCRIPT, "tiktok")
    hook_checks = result["categories"]["hook"]["checks"]
    no_greeting_check = next(c for c in hook_checks if c["name"] == "no_greeting")
    assert no_greeting_check["passed"] is False
    assert no_greeting_check["scored"] == 0


def test_hook_first_sentence_punchy():
    """'I built seven products this year.' is 6 words — should pass ≤8 check."""
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok")
    hook_checks = result["categories"]["hook"]["checks"]
    punchy_check = next(c for c in hook_checks if c["name"] == "first_sentence_punchy")
    assert punchy_check["passed"] is True


def test_contrast_marker_detected():
    """Sample script has no contrast marker in first 2 sentences."""
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok")
    hook_checks = result["categories"]["hook"]["checks"]
    contrast_check = next(c for c in hook_checks if c["name"] == "contrast_marker")
    # "I built seven products" + "Without a team" — no contrast markers
    assert contrast_check["passed"] is False


def test_wpm_estimate_tiktok():
    """Word count / TikTok target WPM midpoint (190) * 60 = estimated duration."""
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok")
    wc = result["word_count"]
    expected_duration = round((wc / 200) * 60, 1)  # TikTok midpoint is 200
    assert abs(result["estimated_duration_seconds"] - expected_duration) < 1


def test_cta_positioned_end():
    """'Follow for more' at the end should be detected as CTA in last 20%."""
    gemini_result = {"sections_detected": ["problem", "solution", "cta"], "focus_score": "single_thread"}
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", gemini_result)
    structure_checks = result["categories"]["structure"]["checks"]
    cta_check = next(c for c in structure_checks if c["name"] == "cta_positioned")
    assert cta_check["passed"] is True


def test_empty_script_handled():
    """Empty script should still return a valid scorecard."""
    result = compute_script_scorecard("", "tiktok")
    assert result["grade"] is not None
    assert result["word_count"] == 0


def test_platform_targets_vary():
    """Different platforms should produce different duration estimates."""
    tiktok = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok")
    linkedin = compute_script_scorecard(SAMPLE_SCRIPT, "linkedin")
    # LinkedIn has lower WPM target → longer estimated duration
    assert linkedin["estimated_duration_seconds"] > tiktok["estimated_duration_seconds"]


def test_gemini_focus_scoring():
    """Gemini focus_score maps to correct scoring."""
    focused = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", {"sections_detected": [], "focus_score": "single_thread"})
    unfocused = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", {"sections_detected": [], "focus_score": "unfocused"})

    focus_cats_focused = focused["categories"]["focus"]
    focus_cats_unfocused = unfocused["categories"]["focus"]

    assert focus_cats_focused["scored"] == 20  # 15 + 5
    assert focus_cats_unfocused["scored"] == 0


def test_cta_short_script_detected():
    """CTA in last sentence of a short script should be detected (sentence-based, not char-slice)."""
    result = compute_script_scorecard(SHORT_SCRIPT, "tiktok")
    structure_checks = result["categories"]["structure"]["checks"]
    cta_check = next(c for c in structure_checks if c["name"] == "cta_positioned")
    # "Subscribe now" is the last sentence — should pass
    assert cta_check["passed"] is True


def test_cta_at_start_penalized():
    """CTA in first sentence should be penalized."""
    script = "Subscribe now. Then I'll tell you about building products."
    result = compute_script_scorecard(script, "tiktok")
    structure_checks = result["categories"]["structure"]["checks"]
    no_early_check = next(c for c in structure_checks if c["name"] == "no_early_cta")
    assert no_early_check["passed"] is False


def test_grade_renormalization():
    """Grade should be based on percentage of available points, not total 100."""
    # Without Gemini, max is 65. If we get 55/65, that's 84.6% = A
    # With a well-structured script that passes all rule-based checks
    result = compute_script_scorecard(SAMPLE_SCRIPT, "tiktok", gemini_result=None)
    # The grade should be based on scored/max_possible, not scored/100
    assert result["percentage"] == round((result["total"] / result["max_possible"]) * 100, 1)
