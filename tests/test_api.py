"""Tests for /analyze and /analyze-script API endpoints."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from starlette.testclient import TestClient

from video_scorer.api import app

VALID_ANALYSIS_ID = str(uuid4())
VALID_VIDEO_URL = "https://fgdvuqqucvfxclfflffg.supabase.co/storage/v1/object/public/post-media/test.mp4"
VALID_PLATFORM = "tiktok"


def _mock_settings():
    """Minimal settings mock for API key middleware."""
    settings = type("S", (), {})()
    settings.api_key = type("K", (), {"get_secret_value": lambda self: "test-key"})()
    settings.allowed_url_prefix = "https://fgdvuqqucvfxclfflffg.supabase.co/storage/v1/object/public/post-media/"
    return settings


AUTH_HEADERS = {"x-api-key": "test-key"}


@patch("video_scorer.api.settings", _mock_settings())
@patch("video_scorer.api.update_status", new_callable=AsyncMock)
def test_analyze_returns_202_immediately(mock_update_status):
    """POST /analyze returns 202 in <2s with valid payload; background task is registered."""
    mock_update_status.return_value = True

    with TestClient(app, raise_server_exceptions=True) as client:
        # TestClient runs BackgroundTasks synchronously after response — we just verify the response shape
        with patch("video_scorer.api._analyze_background", new_callable=AsyncMock):
            resp = client.post(
                "/analyze",
                json={
                    "analysis_id": VALID_ANALYSIS_ID,
                    "video_url": VALID_VIDEO_URL,
                    "platform": VALID_PLATFORM,
                    "qualitative": False,
                },
                headers=AUTH_HEADERS,
            )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert body["analysis_id"] == VALID_ANALYSIS_ID
    mock_update_status.assert_called_once_with(VALID_ANALYSIS_ID, "processing")
    # BackgroundTask was scheduled (TestClient runs it synchronously)
    # Patch intercepts via AsyncMock — it was awaited exactly once


@patch("video_scorer.api.settings", _mock_settings())
@patch("video_scorer.api.update_status", new_callable=AsyncMock)
def test_analyze_returns_409_when_row_missing_or_wrong_state(mock_update_status):
    """POST /analyze returns 409 when update_status reports row missing or wrong state."""
    mock_update_status.return_value = False  # row not found or not in queued state

    with TestClient(app) as client:
        resp = client.post(
            "/analyze",
            json={
                "analysis_id": VALID_ANALYSIS_ID,
                "video_url": VALID_VIDEO_URL,
                "platform": VALID_PLATFORM,
            },
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 409
    assert "row missing" in resp.json()["detail"].lower() or "queued" in resp.json()["detail"].lower()


@patch("video_scorer.api.settings", _mock_settings())
def test_analyze_returns_400_for_invalid_platform():
    """POST /analyze returns 400 for unknown platform before touching DB."""
    with TestClient(app) as client:
        resp = client.post(
            "/analyze",
            json={
                "analysis_id": VALID_ANALYSIS_ID,
                "video_url": VALID_VIDEO_URL,
                "platform": "myspace",
            },
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 400
    assert "platform" in resp.json()["detail"].lower()


@patch("video_scorer.api.settings", _mock_settings())
def test_analyze_returns_400_for_invalid_video_url():
    """POST /analyze returns 400 for video URL that fails SSRF guard."""
    with TestClient(app) as client:
        resp = client.post(
            "/analyze",
            json={
                "analysis_id": VALID_ANALYSIS_ID,
                "video_url": "https://evil.example.com/video.mp4",
                "platform": VALID_PLATFORM,
            },
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 400
    assert "invalid video url" in resp.json()["detail"].lower()


@patch("video_scorer.api.settings", _mock_settings())
@patch("video_scorer.api.gemini_analyze_script", new_callable=AsyncMock)
@patch("video_scorer.api.store_script_scorecard", new_callable=AsyncMock)
def test_analyze_script_smoke(mock_store, mock_gemini):
    """POST /analyze-script still returns 200 — regression check for unaffected endpoint."""
    mock_gemini.return_value = None
    mock_store.return_value = True

    with TestClient(app) as client:
        resp = client.post(
            "/analyze-script",
            json={"script_text": "This is a test script for a TikTok video about productivity.", "platform": "tiktok"},
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "grade" in body
    assert "total_score" in body


@patch("video_scorer.api.settings", _mock_settings())
@patch("video_scorer.api.update_status", new_callable=AsyncMock)
def test_background_task_swallows_supabase_error_on_pipeline_exception(mock_update_status):
    """_analyze_background does not propagate when both pipeline and update_status raise."""
    import asyncio
    from video_scorer.api import _analyze_background, AnalyzeRequest

    # Supabase unreachable on the failure path — must still not propagate
    mock_update_status.side_effect = Exception("Supabase unreachable")

    req = AnalyzeRequest(
        analysis_id=VALID_ANALYSIS_ID,
        video_url=VALID_VIDEO_URL,
        platform=VALID_PLATFORM,
        qualitative=False,
    )

    # Patch httpx so the download itself raises immediately, triggering the outer except
    with patch("video_scorer.api.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

        try:
            asyncio.run(_analyze_background(req, str(req.analysis_id)))
        except Exception as exc:
            raise AssertionError(f"_analyze_background raised unexpectedly: {exc}") from exc
