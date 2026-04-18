"""FastAPI wrapper for the video-scorer pipeline."""

import asyncio
import hmac
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import UUID

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from video_scorer.cli import _run_pipeline
from video_scorer.config import settings
from video_scorer.qualitative.gemini import analyze_script as gemini_analyze_script, analyze_comparison as gemini_compare
from video_scorer.voice_example import generate_voice_example_markdown
from video_scorer.scoring.script_scorecard import compute_script_scorecard
from video_scorer.storage.supabase import store_scorecard, store_script_scorecard, update_status, _sanitize_text

app = FastAPI(title="Video Scorer API", version="0.1.0")

VALID_PLATFORMS = {"tiktok", "youtube", "instagram", "linkedin", "x"}


class AnalyzeRequest(BaseModel):
    analysis_id: UUID
    video_url: str
    platform: str = "tiktok"
    qualitative: bool = False


class ScriptAnalyzeRequest(BaseModel):
    script_text: str
    platform: str = "tiktok"


class CompareRequest(BaseModel):
    ai_draft_text: str | None = None
    final_script_text: str
    spoken_transcript: str
    platform: str = "tiktok"
    topic: str | None = None

MAX_SCRIPT_CHARS = 50_000


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if not settings.api_key:
        return JSONResponse(status_code=503, content={"detail": "API key not configured"})
    provided = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(provided, settings.api_key.get_secret_value()):
        return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/compare-scripts")
async def compare_scripts(req: CompareRequest):
    """Synchronous three-way comparison — AI draft vs final script vs spoken transcript."""
    if req.platform not in VALID_PLATFORMS:
        return JSONResponse(status_code=400, content={"detail": f"Invalid platform: {req.platform}"})

    if not req.final_script_text or not req.final_script_text.strip():
        return JSONResponse(status_code=400, content={"detail": "Final script text is required"})

    if not req.spoken_transcript or not req.spoken_transcript.strip():
        return JSONResponse(status_code=400, content={"detail": "Spoken transcript is required"})

    if len(req.final_script_text) > MAX_SCRIPT_CHARS or len(req.spoken_transcript) > MAX_SCRIPT_CHARS:
        return JSONResponse(status_code=400, content={"detail": f"Text too long. Maximum {MAX_SCRIPT_CHARS} characters per field."})

    if req.ai_draft_text and len(req.ai_draft_text) > MAX_SCRIPT_CHARS:
        return JSONResponse(status_code=400, content={"detail": f"AI draft too long. Maximum {MAX_SCRIPT_CHARS} characters."})

    # Run Gemini comparison analysis
    delta_analysis = await gemini_compare(
        ai_draft_text=req.ai_draft_text,
        final_script_text=req.final_script_text,
        spoken_transcript=req.spoken_transcript,
        platform=req.platform,
    )

    if not delta_analysis:
        return JSONResponse(status_code=502, content={"detail": "Comparison analysis failed. Gemini may be unavailable."})

    # Generate voice example markdown
    voice_md, voice_filename = generate_voice_example_markdown(
        platform=req.platform,
        ai_draft_text=req.ai_draft_text,
        final_script_text=req.final_script_text,
        spoken_transcript=req.spoken_transcript,
        delta_analysis=delta_analysis,
        topic=req.topic,
    )

    return {
        "delta_analysis": delta_analysis,
        "voice_example_content": voice_md,
        "voice_example_filename": voice_filename,
        "platform": req.platform,
    }


@app.post("/analyze-script")
async def analyze_script(req: ScriptAnalyzeRequest):
    """Synchronous script analysis — rule-based checks + Gemini qualitative."""
    # Validate platform
    if req.platform not in VALID_PLATFORMS:
        return JSONResponse(status_code=400, content={"detail": f"Invalid platform: {req.platform}"})

    # Validate script text
    if not req.script_text or not req.script_text.strip():
        return JSONResponse(status_code=400, content={"detail": "Script text is required"})

    if len(req.script_text) > MAX_SCRIPT_CHARS:
        return JSONResponse(status_code=400, content={"detail": f"Script too long. Maximum {MAX_SCRIPT_CHARS} characters."})

    warnings = []

    # Step 1: Compute rule-based scorecard (without Gemini)
    scorecard = compute_script_scorecard(req.script_text, req.platform)

    # Step 2: Run Gemini qualitative analysis (always attempted)
    gemini_result = await gemini_analyze_script(
        script_text=req.script_text,
        platform=req.platform,
        word_count=scorecard["word_count"],
        estimated_duration=scorecard["estimated_duration_seconds"],
        target_wpm=scorecard["target_wpm"],
    )

    if gemini_result:
        # Recompute scorecard with Gemini results (for focus + structure.sections)
        scorecard = compute_script_scorecard(req.script_text, req.platform, gemini_result)
    else:
        warnings.append("Gemini qualitative analysis unavailable. Scoring based on rule-based checks only (max 65 points).")

    # Step 3: Build response
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))

    result = {
        "script_text": req.script_text,
        "platform": req.platform,
        "grade": scorecard["grade"],
        "total_score": scorecard["total"],
        "max_possible": scorecard["max_possible"],
        "percentage": scorecard["percentage"],
        "word_count": scorecard["word_count"],
        "estimated_duration_seconds": scorecard["estimated_duration_seconds"],
        "target_wpm": scorecard["target_wpm"],
        "breakdown": scorecard["breakdown"],
        "categories": scorecard["categories"],
        "platform_targets": scorecard["platform_targets"],
        "qualitative": gemini_result,
        "raw_scorecard": scorecard,
        "warnings": warnings,
        "scored_at": now_et.isoformat(),
    }

    # Step 4: Store in Supabase
    stored = await store_script_scorecard(result)
    if not stored:
        warnings.append("Failed to store scorecard in database.")

    return result


async def _analyze_background(req: AnalyzeRequest, analysis_id: str) -> None:
    """Run the full video pipeline in the background after /analyze returns 202."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / _filename_from_url(req.video_url)

            async with httpx.AsyncClient(timeout=300, follow_redirects=False) as client:
                async with client.stream("GET", req.video_url) as resp:
                    resp.raise_for_status()
                    with open(video_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            f.write(chunk)

            # _run_pipeline is async def but contains blocking FFmpeg subprocess calls.
            # asyncio.to_thread keeps those blocking calls off the event loop so subsequent
            # /analyze dispatches are not stalled on this container's single uvicorn worker.
            scorecard = await asyncio.to_thread(
                asyncio.run,
                _run_pipeline(
                    path=video_path,
                    platform=req.platform,
                    qualitative=req.qualitative,
                    store=False,
                ),
            )

            stored = await store_scorecard(scorecard, analysis_id)
            if not stored:
                try:
                    await update_status(analysis_id, "failed", "Failed to store scorecard in database")
                except Exception as supabase_err:
                    print(f"  Warning: update_status failed after store failure: {supabase_err}", file=sys.stderr)

    except Exception as e:
        error_msg = f"Pipeline error: {type(e).__name__}: {str(e)[:200]}"
        print(f"  Background task error for analysis_id={analysis_id}: {error_msg}", file=sys.stderr)
        try:
            await update_status(analysis_id, "failed", error_msg)
        except Exception as supabase_err:
            print(f"  Warning: update_status failed after pipeline error: {supabase_err}", file=sys.stderr)


@app.post("/analyze", status_code=202)
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    # Validate platform
    if req.platform not in VALID_PLATFORMS:
        return JSONResponse(status_code=400, content={"detail": f"Invalid platform: {req.platform}"})

    # SSRF guard — validate URL host and path prefix (decode %2e%2e etc.)
    parsed = urlparse(req.video_url)
    expected = urlparse(settings.allowed_url_prefix)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected.hostname
        or not decoded_path.startswith(expected.path)
        or ".." in decoded_path
    ):
        return JSONResponse(status_code=400, content={"detail": "Invalid video URL"})

    analysis_id = str(req.analysis_id)

    # Claim the row — abort if row doesn't exist or isn't in queued state
    status_updated = await update_status(analysis_id, "processing")
    if not status_updated:
        return JSONResponse(
            status_code=409,
            content={"detail": f"Cannot process analysis_id={analysis_id}: row missing or not in queued state"},
        )

    background_tasks.add_task(_analyze_background, req, analysis_id)
    return {"status": "processing", "analysis_id": analysis_id}


def _filename_from_url(url: str) -> str:
    """Extract filename from Supabase Storage URL."""
    path = url.split("/")[-1].split("?")[0]
    if not path or "\x00" in path:
        return "video.mp4"
    return path.replace("/", "_")
