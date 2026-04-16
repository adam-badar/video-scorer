"""Optional Gemini 2.5 Pro qualitative analysis for hook quality and content assessment."""

import json
import sys

import httpx

from video_scorer.config import settings

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

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
    """Strip prompt boundary markers from text before interpolation into a prompt template.

    Case-insensitive to prevent bypass via &lt;/SCRIPT&gt;, &lt;Script&gt;, etc.
    """
    import re
    result = text
    for tag in boundary_tags:
        result = re.sub(re.escape(tag), "", result, flags=re.IGNORECASE)
    return result


VALID_SECTIONS = {"problem", "solution", "proof", "cta"}
VALID_FOCUS_SCORES = {"single_thread", "branches", "unfocused"}


def _validate_script_gemini_output(result: dict) -> dict:
    """Validate and clamp Gemini script analysis output to prevent score manipulation.

    Untrusted Gemini output could inflate scores via prompt injection.
    Only allow known values for scoring-critical fields.
    """
    # Validate sections_detected — only allow known section names, normalize to lowercase
    sections = result.get("sections_detected", [])
    if isinstance(sections, list):
        result["sections_detected"] = [s.lower() for s in sections if isinstance(s, str) and s.lower() in VALID_SECTIONS]
    else:
        result["sections_detected"] = []

    # Validate focus_score — only allow known values
    focus = result.get("focus_score", "")
    if not isinstance(focus, str) or focus.lower() not in VALID_FOCUS_SCORES:
        result["focus_score"] = "unfocused"  # Conservative default
    else:
        result["focus_score"] = focus.lower()

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
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 1024},
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
        parsed = json.loads(text)
        return _validate_script_gemini_output(parsed)

    except json.JSONDecodeError:
        print(f"  Warning: Gemini returned non-JSON for script analysis: {text[:200]}", file=sys.stderr)
        return None
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Gemini script analysis failed: {e}", file=sys.stderr)
        return None


COMPARE_SYSTEM_PROMPT = """You are a content voice analyst. You compare three versions of the same content: the AI-generated draft, the human-edited final script, and what was actually spoken on camera. Your job is to extract patterns that will make future AI drafts sound more like how the creator actually speaks."""

COMPARE_PROMPT_TEMPLATE = """Compare these three versions of the same video content.

<ai_draft>
{ai_draft_text}
</ai_draft>

<final_script>
{final_script_text}
</final_script>

<spoken_transcript>
{spoken_transcript}
</spoken_transcript>

<context>
Platform: {platform}
</context>

Analyze the differences between all three versions. Return JSON:
{{
  "deltas": [
    {{
      "what_changed": "Specific text that was different between versions",
      "from_version": "ai_draft | final_script",
      "to_version": "final_script | spoken_transcript",
      "why": "Why this change was made (inferred from context)",
      "pattern_to_carry_forward": "A reusable rule for future AI drafts"
    }}
  ],
  "voice_patterns": [
    "Concise patterns about how the creator speaks vs writes — e.g., 'Uses shorter sentences on camera than in script', 'Drops technical jargon when speaking'"
  ],
  "overall_assessment": "2-3 sentences summarizing the key differences and what the AI draft should do differently next time"
}}

Focus on patterns that are reusable — not one-off edits. Include at least 3 deltas and 2 voice patterns.

Respond with ONLY the JSON object, no markdown formatting."""

COMPARE_BOUNDARY_TAGS = ["<ai_draft>", "</ai_draft>", "<final_script>", "</final_script>", "<spoken_transcript>", "</spoken_transcript>", "<context>", "</context>"]


async def analyze_comparison(
    ai_draft_text: str | None,
    final_script_text: str,
    spoken_transcript: str,
    platform: str,
) -> dict | None:
    """Generate three-way delta analysis via Gemini.

    Returns structured delta dict, or None on failure.
    """
    if not settings.gemini_api_key:
        print("  Warning: Set VIDEO_SCORER_GEMINI_API_KEY for comparison analysis.", file=sys.stderr)
        return None

    if not final_script_text.strip() or not spoken_transcript.strip():
        print("  Skipped: Need both final script and spoken transcript.", file=sys.stderr)
        return None

    # Sanitize all text fields
    safe_draft = _sanitize_for_prompt(ai_draft_text or "", COMPARE_BOUNDARY_TAGS + SCRIPT_BOUNDARY_TAGS)
    safe_final = _sanitize_for_prompt(final_script_text, COMPARE_BOUNDARY_TAGS + SCRIPT_BOUNDARY_TAGS)
    safe_spoken = _sanitize_for_prompt(spoken_transcript, COMPARE_BOUNDARY_TAGS + VIDEO_BOUNDARY_TAGS)

    # Handle two-way comparison when AI draft is empty
    draft_text = safe_draft if safe_draft.strip() else "(No AI draft provided — compare final script to spoken transcript only)"

    prompt = COMPARE_PROMPT_TEMPLATE.format(
        ai_draft_text=draft_text,
        final_script_text=safe_final,
        spoken_transcript=safe_spoken,
        platform=platform,
    )

    payload = {
        "system_instruction": {"parts": [{"text": COMPARE_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 1024},
        },
    }

    url = f"{GEMINI_URL}?key={settings.gemini_api_key.get_secret_value().strip()}"

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            print(f"  Warning: Gemini comparison analysis failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
            return None

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            print("  Warning: Gemini returned no candidates for comparison.", file=sys.stderr)
            return None

        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = json.loads(text)

        # Validate structure — each delta must be a dict
        if "deltas" not in parsed or not isinstance(parsed["deltas"], list):
            print("  Warning: Gemini comparison missing 'deltas' array.", file=sys.stderr)
            return None

        validated_deltas = []
        for d in parsed["deltas"]:
            if not isinstance(d, dict):
                continue
            validated_deltas.append({
                "what_changed": str(d.get("what_changed", "")),
                "from_version": str(d.get("from_version", "")),
                "to_version": str(d.get("to_version", "")),
                "why": str(d.get("why", "")),
                "pattern_to_carry_forward": str(d.get("pattern_to_carry_forward", "")),
            })
        parsed["deltas"] = validated_deltas

        patterns = parsed.get("voice_patterns", [])
        parsed["voice_patterns"] = [str(p) for p in patterns if isinstance(p, (str, int, float))] if isinstance(patterns, list) else []

        return parsed

    except json.JSONDecodeError:
        print(f"  Warning: Gemini returned non-JSON for comparison: {text[:200]}", file=sys.stderr)
        return None
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Gemini comparison analysis failed: {e}", file=sys.stderr)
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
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 1024},
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
