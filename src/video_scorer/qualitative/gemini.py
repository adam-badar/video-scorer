"""Optional Gemini 2.5 Pro qualitative analysis for hook quality and content assessment."""

import json
import sys

import httpx

from video_scorer.config import settings

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-05-06:generateContent"

SYSTEM_PROMPT = """You are a short-form video content analyst specializing in TikTok, YouTube Shorts, Instagram Reels, and LinkedIn video. You analyze video transcripts and metrics to provide actionable feedback.

Your feedback must be specific and quantified — not generic advice. Reference the exact metrics provided."""

SCRIPT_SYSTEM_PROMPT = """You are a short-form video script analyst specializing in TikTok, YouTube Shorts, Instagram Reels, and LinkedIn video. You analyze scripts before filming to provide actionable feedback on hook quality, narrative structure, focus, and pacing.

Your feedback must be specific — reference the exact text in the script. Do not give generic advice."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this short-form video for content quality and optimization potential.

<video_metadata>
Platform: {platform}
Duration: {duration}s
Resolution: {resolution}
</video_metadata>

<metrics>
WPM: {wpm}
Filler words: {filler_count}
Cuts per minute: {cuts_per_minute}
Loudness: {loudness_lufs} LUFS
Loop score: {loop_score}
Current grade: {grade} ({total}/{max_possible} = {percentage}%)
</metrics>

<transcript>
{transcript}
</transcript>

Provide analysis in this exact JSON structure:
{{
  "hook_assessment": "1-2 sentences on the first 3 seconds — is the hook strong? What specific improvement would you make?",
  "pacing_assessment": "1-2 sentences on speaking pace and flow",
  "structure_assessment": "1-2 sentences on the narrative arc (problem-solution-proof-CTA)",
  "top_improvement": "The single highest-leverage change to improve this video's performance",
  "estimated_grade_after_fix": "What grade this video could reach if the top improvement is applied"
}}

Respond with ONLY the JSON object, no markdown formatting."""


SCRIPT_ANALYSIS_PROMPT_TEMPLATE = """Analyze this script for short-form video pre-filming.

<script_metadata>
Platform: {platform}
Word count: {word_count}
Estimated duration: {estimated_duration}s (at {target_wpm} WPM target)
</script_metadata>

<script>
{script_text}
</script>

Provide analysis in this exact JSON structure:
{{
  "hook_assessment": "1-2 sentences on the opening — is the hook strong? What specific improvement would you make?",
  "pacing_assessment": "1-2 sentences on estimated pacing and flow",
  "structure_assessment": "1-2 sentences on the narrative arc (problem-solution-proof-CTA)",
  "narrative_focus": "Does this script tell one story? If it branches, what should be cut?",
  "top_improvement": "The single highest-leverage change to improve this script",
  "estimated_grade_after_fix": "What grade this script could reach if the top improvement is applied",
  "sections_detected": ["problem", "solution", "proof", "cta"],
  "focus_score": "single_thread | branches | unfocused"
}}

For sections_detected, only include sections that are actually present in the script.
For focus_score, choose exactly one of: single_thread, branches, unfocused.

Respond with ONLY the JSON object, no markdown formatting."""


def _sanitize_for_prompt(text: str, boundary_tags: list[str]) -> str:
    """Strip prompt boundary markers from text before interpolation into a prompt template."""
    result = text
    for tag in boundary_tags:
        result = result.replace(tag, "")
    return result


# Boundary tags for each prompt type
VIDEO_BOUNDARY_TAGS = ["<transcript>", "</transcript>", "<video_metadata>", "</video_metadata>", "<metrics>", "</metrics>"]
SCRIPT_BOUNDARY_TAGS = ["<script>", "</script>", "<script_metadata>", "</script_metadata>"]


async def analyze_script(script_text: str, platform: str, word_count: int, estimated_duration: float, target_wpm: float) -> dict | None:
    """Send script to Gemini for qualitative pre-filming analysis.

    Returns parsed analysis dict, or None on failure.
    """
    if not settings.gemini_api_key:
        print("  Warning: Set VIDEO_SCORER_GEMINI_API_KEY for script analysis.", file=sys.stderr)
        return None

    if not script_text or len(script_text.strip()) < 10:
        print("  Skipped: Script too short for qualitative analysis.", file=sys.stderr)
        return None

    # Sanitize script text — strip prompt boundary markers
    safe_script = _sanitize_for_prompt(script_text, SCRIPT_BOUNDARY_TAGS)

    # Truncate if too long
    estimated_tokens = len(safe_script) / 4
    if estimated_tokens > 100_000:
        chars = len(safe_script)
        keep = int(chars * 0.2)
        safe_script = safe_script[:keep] + "\n\n[...middle truncated...]\n\n" + safe_script[-keep:]
        print(f"  Script truncated from {chars} to {len(safe_script)} chars for Gemini.", file=sys.stderr)

    prompt = SCRIPT_ANALYSIS_PROMPT_TEMPLATE.format(
        platform=platform,
        word_count=word_count,
        estimated_duration=estimated_duration,
        target_wpm=target_wpm,
        script_text=safe_script,
    )

    payload = {
        "system_instruction": {"parts": [{"text": SCRIPT_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }

    url = f"{GEMINI_URL}?key={settings.gemini_api_key.get_secret_value().strip()}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            print(f"  Warning: Gemini script analysis failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
            return None

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            print("  Warning: Gemini returned no candidates for script analysis.", file=sys.stderr)
            return None

        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return json.loads(text)

    except json.JSONDecodeError:
        print(f"  Warning: Gemini returned non-JSON for script analysis: {text[:200]}", file=sys.stderr)
        return None
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Gemini script analysis failed: {e}", file=sys.stderr)
        return None


async def analyze(scorecard: dict, transcript: str) -> dict | None:
    """Send transcript + metrics to Gemini for qualitative assessment.

    Returns parsed analysis dict, or None on failure.
    Note: This sends transcript text to Google's API. Google AI Studio terms
    may allow input use for model improvement. For privacy-sensitive content,
    use Vertex AI with a DPA instead.
    """
    if not settings.gemini_api_key:
        print("  Warning: Set VIDEO_SCORER_GEMINI_API_KEY for qualitative analysis.", file=sys.stderr)
        return None

    if not transcript or len(transcript.strip()) < 10:
        print("  Skipped: No transcript available for qualitative analysis.", file=sys.stderr)
        return None

    # Sanitize transcript — strip potential prompt boundary markers
    safe_transcript = _sanitize_for_prompt(transcript, VIDEO_BOUNDARY_TAGS)

    # Check token budget (~4 chars per token rough estimate)
    estimated_tokens = len(safe_transcript) / 4
    if estimated_tokens > 100_000:
        # Truncate: keep first 20% + last 20%
        chars = len(safe_transcript)
        keep = int(chars * 0.2)
        safe_transcript = safe_transcript[:keep] + "\n\n[...middle truncated...]\n\n" + safe_transcript[-keep:]
        print(f"  Transcript truncated from {chars} to {len(safe_transcript)} chars for Gemini.", file=sys.stderr)

    video = scorecard.get("video", {})
    pacing = scorecard.get("pacing", {})
    editing = scorecard.get("editing", {})
    audio = scorecard.get("audio", {})
    score = scorecard.get("score", {})

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        platform=score.get("platform_targets", {}).get("platform", "tiktok"),
        duration=video.get("duration_seconds", "?"),
        resolution=video.get("resolution", "?"),
        wpm=pacing.get("wpm", "N/A"),
        filler_count=pacing.get("filler_word_count", "N/A"),
        cuts_per_minute=editing.get("cuts_per_minute", "N/A"),
        loudness_lufs=audio.get("loudness_lufs", "N/A"),
        loop_score=scorecard.get("structure", {}).get("loop_score", "N/A"),
        grade=score.get("grade", "?"),
        total=score.get("total", "?"),
        max_possible=score.get("max_possible", "?"),
        percentage=score.get("percentage", "?"),
        transcript=safe_transcript,
    )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }

    url = f"{GEMINI_URL}?key={settings.gemini_api_key.get_secret_value().strip()}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            print(f"  Warning: Gemini analysis failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
            return None

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            print("  Warning: Gemini returned no candidates.", file=sys.stderr)
            return None

        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return json.loads(text)

    except json.JSONDecodeError:
        print(f"  Warning: Gemini returned non-JSON response: {text[:200]}", file=sys.stderr)
        return None
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Gemini analysis failed: {e}", file=sys.stderr)
        return None
