"""Store scorecards in Supabase via REST API (no SDK dependency)."""

import json
import sys

import httpx

from video_scorer.config import settings


async def store_scorecard(scorecard: dict) -> bool:
    """Upsert scorecard to Supabase video_scorecards table.

    Uses file_hash as the idempotency key (UNIQUE constraint).
    Returns True on success, False on failure (logs warning, never raises).
    """
    if not settings.supabase_url or not settings.supabase_anon_key:
        print("  Warning: Set VIDEO_SCORER_SUPABASE_URL and VIDEO_SCORER_SUPABASE_ANON_KEY to store results.", file=sys.stderr)
        return False

    video = scorecard.get("video", {})
    pacing = scorecard.get("pacing", {})
    editing = scorecard.get("editing", {})
    audio = scorecard.get("audio", {})
    score = scorecard.get("score", {})

    row = {
        "file_hash": video.get("file_hash"),
        "file_name": video.get("file"),
        "platform": score.get("platform_targets", {}).get("platform", "tiktok"),
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
        "qualitative": scorecard.get("qualitative"),
        "raw_scorecard": scorecard,
        "scored_at": scorecard.get("scored_at"),
    }

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/video_scorecards"
    headers = {
        "apikey": settings.supabase_anon_key.get_secret_value().strip(),
        "Authorization": f"Bearer {settings.supabase_anon_key.get_secret_value().strip()}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers=headers,
                content=json.dumps(row, default=str),
            )

        if resp.status_code in (200, 201):
            return True

        # Supabase returns 409 for conflict without Prefer header, but with
        # resolution=merge-duplicates it should upsert. Log any other error.
        print(f"  Warning: Supabase storage failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return False

    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        print(f"  Warning: Supabase storage failed: {e}", file=sys.stderr)
        return False
