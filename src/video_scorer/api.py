"""FastAPI wrapper for the video-scorer pipeline."""

import asyncio
import hmac
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from video_scorer.cli import _run_pipeline
from video_scorer.config import settings
from video_scorer.storage.supabase import store_scorecard, update_status

app = FastAPI(title="Video Scorer API", version="0.1.0")

VALID_PLATFORMS = {"tiktok", "youtube", "instagram", "linkedin"}


class AnalyzeRequest(BaseModel):
    analysis_id: UUID
    video_url: str
    platform: str = "tiktok"
    qualitative: bool = False


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="API key not configured")
    provided = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(provided, settings.api_key.get_secret_value()):
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

    # SSRF guard — validate URL host and path prefix
    parsed = urlparse(req.video_url)
    expected = urlparse(settings.allowed_url_prefix)
    if parsed.scheme != "https" or parsed.hostname != expected.hostname or not parsed.path.startswith(expected.path):
        raise HTTPException(status_code=400, detail="Invalid video URL: must be a Supabase Storage URL")
    # Reject path traversal
    if ".." in parsed.path:
        raise HTTPException(status_code=400, detail="Invalid video URL: path traversal not allowed")

    analysis_id = str(req.analysis_id)

    # Update status to processing
    await update_status(analysis_id, "processing")

    try:
        # Download video to temp file using streaming (avoids loading full file into memory)
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / _filename_from_url(req.video_url)

            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                async with client.stream("GET", req.video_url) as resp:
                    resp.raise_for_status()
                    with open(video_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            f.write(chunk)

            # Run pipeline in a thread to avoid blocking the async event loop
            # (_run_pipeline uses subprocess.run internally for FFmpeg calls)
            scorecard = await asyncio.to_thread(
                asyncio.run,
                _run_pipeline(
                    path=video_path,
                    platform=req.platform,
                    qualitative=req.qualitative,
                    store=False,
                ),
            )

            # Store result by analysis_id
            stored = await store_scorecard(scorecard, analysis_id)
            if not stored:
                await update_status(analysis_id, "failed", "Failed to store scorecard in database")
                return {"status": "failed"}

    except httpx.HTTPStatusError as e:
        error_msg = f"Failed to download video: {e.response.status_code}"
        await update_status(analysis_id, "failed", error_msg)
        return {"status": "failed"}
    except Exception as e:
        error_msg = f"Pipeline error: {type(e).__name__}: {str(e)[:200]}"
        await update_status(analysis_id, "failed", error_msg)
        return {"status": "failed"}

    return {"status": "succeeded"}


def _filename_from_url(url: str) -> str:
    """Extract filename from Supabase Storage URL."""
    path = url.split("/")[-1].split("?")[0]
    if not path:
        return "video.mp4"
    return path
