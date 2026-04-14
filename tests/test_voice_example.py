"""Tests for voice example markdown generation and path traversal prevention."""

from video_scorer.voice_example import sanitize_topic_slug, generate_voice_example_markdown


def test_sanitize_normal_topic():
    slug = sanitize_topic_slug("Building Products Solo")
    assert slug == "building-products-solo"


def test_sanitize_path_traversal():
    """Path traversal attempt must be stripped to safe slug."""
    slug = sanitize_topic_slug("../../shared/voice-core")
    assert ".." not in slug
    assert "/" not in slug
    assert slug == "sharedvoice-core"


def test_sanitize_special_chars():
    slug = sanitize_topic_slug("AI & ML: The Future?!")
    # Special chars stripped, spaces become hyphens
    assert "/" not in slug
    assert "?" not in slug
    assert "!" not in slug
    assert slug.startswith("ai")


def test_sanitize_max_length():
    long_topic = "a" * 100
    slug = sanitize_topic_slug(long_topic)
    assert len(slug) <= 50


def test_sanitize_empty():
    slug = sanitize_topic_slug("")
    assert slug == "untitled"


def test_generate_voice_example_three_way():
    """Full three-way comparison produces valid markdown."""
    delta = {
        "deltas": [
            {
                "what_changed": "Removed technical jargon",
                "from_version": "ai_draft",
                "to_version": "final_script",
                "why": "Too complex for TikTok audience",
                "pattern_to_carry_forward": "Use simple language on TikTok",
            }
        ],
        "voice_patterns": ["Shorter sentences on camera", "Drops qualifiers"],
        "overall_assessment": "The creator simplifies significantly between draft and camera.",
    }

    md, filename = generate_voice_example_markdown(
        platform="tiktok",
        ai_draft_text="This is the AI draft with technical jargon.",
        final_script_text="This is the final script, simplified.",
        spoken_transcript="This is what was actually said on camera.",
        delta_analysis=delta,
        topic="building products",
    )

    assert "tiktok" in filename.lower() or "building-products" in filename
    assert "AI draft" in md
    assert "Final script" in md
    assert "Spoken transcript" in md
    assert "Removed technical jargon" in md
    assert "Shorter sentences on camera" in md
    assert "building-products" in filename


def test_generate_voice_example_two_way():
    """Two-way comparison (no AI draft) still produces valid markdown."""
    delta = {
        "deltas": [{"what_changed": "Spoke faster", "why": "Natural pace", "pattern_to_carry_forward": "Target 200 WPM"}],
        "voice_patterns": ["Natural pace is faster than scripted"],
        "overall_assessment": "Camera delivery is naturally faster.",
    }

    md, filename = generate_voice_example_markdown(
        platform="tiktok",
        ai_draft_text=None,
        final_script_text="The final script.",
        spoken_transcript="What was said.",
        delta_analysis=delta,
    )

    assert "**AI draft:**" not in md  # Should not include AI draft section (context line mentions it generically)
    assert "Final script" in md
    assert "comparison" in filename  # Default topic
