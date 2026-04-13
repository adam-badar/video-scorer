"""Store scorecards in Supabase via REST API (no SDK dependency)."""

import json
import sys
import re

import httpx

from video_scorer.config import settings


def _get_headers() -> dict | None:
    """Build Supabase REST headers using service key."""
    if not settings.supabase_url or not settings.supabase_service_key:
        print("  Warning: Set VIDEO_SCORER_SUPABASE_URL and VIDEO_SCORER_SUPABASE_SERVICE_KEY to store results.", file=sys.stderr)
        return None
    key = settings.supabase_service_key.get_secret_value().strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _sanitize_text(text: str | None) -> str | None:
    """Strip mathematical Unicode and control characters that cause PGRST102."""
    if text is None:
        return None
    # Remove Unicode math symbols (U+2200-U+22FF), control chars (except newline/tab)
    return re.sub(r'[\u2200-\u22ff\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


async def insert_queued_row(analysis_id: str, platform: str, video_url: str | None = None) -> bool:
    """Insert a new queued row for an analysis. Used by CLI --store path."""
    headers = _get_headers()
    if not headers:
        return False

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/video_scorecards"
    headers["Prefer"] = "return=minimal"

    row = {
        "analysis_id": analysis_id,
        "platform": platform,
        "video_url": video_url,
        "status": "queued",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, content=json.dumps(row, default=str))
        if resp.status_code in (200, 201):
            return True
        print(f"  Warning: Insert queued row failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return False
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Insert queued row failed: {e}", file=sys.stderr)
        return False


async def update_status(analysis_id: str, status: str, error_message: str | None = None) -> bool:
    """Update the status of an analysis row by analysis_id."""
    headers = _get_headers()
    if not headers:
        return False

    # Only transition from non-terminal states to prevent overwriting succeeded/failed
    allowed_from = "queued,processing" if status == "processing" else "queued,processing"
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/video_scorecards?analysis_id=eq.{analysis_id}&status=in.({allowed_from})"
    headers["Prefer"] = "return=representation"

    body: dict = {"status": status}
    if error_message:
        body["error_message"] = error_message

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.patch(url, headers=headers, content=json.dumps(body, default=str))
        if resp.status_code == 200:
            rows = resp.json()
            if not rows:
                print(f"  Warning: No row found for analysis_id={analysis_id}", file=sys.stderr)
                return False
            return True
        print(f"  Warning: Status update failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return False
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Status update failed: {e}", file=sys.stderr)
        return False


async def store_scorecard(scorecard: dict, analysis_id: str) -> bool:
    """Update scorecard row by analysis_id with full results.

    Sets status to 'succeeded' and populates all metric fields.
    """
    headers = _get_headers()
    if not headers:
        return False

    video = scorecard.get("video", {})
    pacing = scorecard.get("pacing", {})
    editing = scorecard.get("editing", {})
    audio = scorecard.get("audio", {})
    score = scorecard.get("score", {})

    # Sanitize text fields that may contain problematic Unicode
    qualitative = scorecard.get("qualitative")
    if qualitative and isinstance(qualitative, dict):
        qualitative = {k: _sanitize_text(v) if isinstance(v, str) else v for k, v in qualitative.items()}

    row = {
        "file_hash": video.get("file_hash"),
        "file_name": video.get("file"),
        "platform": score.get("platform_targets", {}).get("platform", "tiktok"),
        "status": "succeeded",
        "duration_seconds": video.get("duration_seconds"),
        "resolution": video.get("resolution"),
        "wpm": pacing.get("wpm"),
        "filler_word_count": pacing.get("filler_word_count"),
        "cuts_per_minute": editing.get("cuts_per_minute"),
        "loudness_lufs": audio.get("loudness_lufs"),
        "true_peak_dbtp": audio.get("true_peak_dbtp"),
        "silence_ratio": audio.get("silence_ratio"),
        "loop_score": scorecard.get("structure", {}).get("loop_score"),
        "hook_score": score.get("breakdown", {}).get("hook"),
        "pacing_score": score.get("breakdown", {}).get("pacing"),
        "editing_score": score.get("breakdown", {}).get("editing"),
        "audio_score": score.get("breakdown", {}).get("audio"),
        "structure_score": score.get("breakdown", {}).get("structure"),
        "total_score": score.get("total"),
        "max_possible": score.get("max_possible"),
        "grade": score.get("grade"),
        "detected_language": pacing.get("detected_language"),
        "language_warning": pacing.get("language_warning"),
        "qualitative": qualitative,
        "raw_scorecard": scorecard,
        "scored_at": scorecard.get("scored_at"),
    }

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/video_scorecards?analysis_id=eq.{analysis_id}"
    headers["Prefer"] = "return=representation"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.patch(
                url,
                headers=headers,
                content=json.dumps(row, default=str),
            )

        if resp.status_code == 200:
            rows = resp.json()
            if not rows:
                print(f"  Warning: No row found for analysis_id={analysis_id}", file=sys.stderr)
                return False
            return True

        print(f"  Warning: Supabase storage failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return False

    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Supabase storage failed: {e}", file=sys.stderr)
        return False
