"""FastAPI wrapper for the video-scorer pipeline."""

import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from video_scorer.cli import _run_pipeline
from video_scorer.config import settings
from video_scorer.storage.supabase import store_scorecard, update_status

app = FastAPI(title="Video Scorer API", version="0.1.0")

VALID_PLATFORMS = {"tiktok", "youtube", "instagram", "linkedin"}


class AnalyzeRequest(BaseModel):
    analysis_id: str
    video_url: str
    platform: str = "tiktok"
    qualitative: bool = False


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if settings.api_key:
        provided = request.headers.get("x-api-key", "")
        if provided != settings.api_key.get_secret_value():
            raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", status_code=202)
async def analyze(req: AnalyzeRequest):
    # Validate platform
    if req.platform not in VALID_PLATFORMS:
        raise HTTPException(status_code=400, detail=f"Invalid platform: {req.platform}")

    # SSRF guard — validate URL prefix
    if not req.video_url.startswith(settings.allowed_url_prefix):
        raise HTTPException(status_code=400, detail="Invalid video URL: must be a Supabase Storage URL")

    # Update status to processing
    await update_status(req.analysis_id, "processing")

    try:
        # Download video to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / _filename_from_url(req.video_url)

            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                resp = await client.get(req.video_url)
                resp.raise_for_status()
                video_path.write_bytes(resp.content)

            # Run the existing pipeline
            scorecard = await _run_pipeline(
                path=video_path,
                platform=req.platform,
                qualitative=req.qualitative,
                store=False,  # We handle storage ourselves with analysis_id
            )

            # Store result by analysis_id
            stored = await store_scorecard(scorecard, req.analysis_id)
            if not stored:
                await update_status(req.analysis_id, "failed", "Failed to store scorecard in database")
                return {"status": "failed"}

    except httpx.HTTPStatusError as e:
        error_msg = f"Failed to download video: {e.response.status_code}"
        await update_status(req.analysis_id, "failed", error_msg)
        return {"status": "failed"}
    except Exception as e:
        error_msg = f"Pipeline error: {type(e).__name__}: {str(e)[:200]}"
        await update_status(req.analysis_id, "failed", error_msg)
        return {"status": "failed"}

    return {"status": "succeeded"}


def _filename_from_url(url: str) -> str:
    """Extract filename from Supabase Storage URL."""
    path = url.split("/")[-1].split("?")[0]
    if not path:
        return "video.mp4"
    return path
