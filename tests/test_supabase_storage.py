"""Tests for Supabase storage (mocked — no real API calls)."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx

from video_scorer.storage.supabase import store_scorecard


def _sample_scorecard():
    return {
        "video": {
            "file": "test.mp4",
            "file_hash": "sha256:abc123",
            "duration_seconds": 45.0,
            "resolution": "1080x1920",
        },
        "pacing": {"wpm": 185.0, "filler_word_count": 0, "detected_language": "en", "language_warning": None},
        "editing": {"cuts_per_minute": 10.0},
        "audio": {"loudness_lufs": -14.2, "true_peak_dbtp": -1.8, "silence_ratio": 0.03},
        "structure": {"loop_score": 0.85},
        "sentiment": None,
        "topics": None,
        "score": {
            "total": 120, "max_possible": 130, "grade": "A", "percentage": 92.3,
            "breakdown": {"hook": 40, "pacing": 25, "editing": 20, "audio": 25, "structure": 10},
            "platform_targets": {"platform": "tiktok"},
        },
        "qualitative": None,
        "scored_at": "2026-04-11T15:30:00-04:00",
    }


@patch("video_scorer.storage.supabase.settings")
def test_store_success(mock_settings):
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = type("S", (), {"get_secret_value": lambda self: "test-key"})()

    mock_resp = AsyncMock()
    mock_resp.status_code = 204

    with patch("video_scorer.storage.supabase.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.patch = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(store_scorecard(_sample_scorecard(), "test-analysis-id"))
        assert result is True

        # Verify PATCH was called with analysis_id in URL
        call_args = mock_client.patch.call_args
        url = call_args[1].get("url", call_args[0][0] if call_args[0] else "")
        assert "video_scorecards" in url
        assert "test-analysis-id" in url
        # Verify the body contains scorecard data
        body = json.loads(call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "{}"))
        assert body["file_hash"] == "sha256:abc123"
        assert body["grade"] == "A"
        assert body["status"] == "succeeded"


@patch("video_scorer.storage.supabase.settings")
def test_store_no_credentials(mock_settings):
    mock_settings.supabase_url = None
    mock_settings.supabase_service_key = None

    result = asyncio.run(store_scorecard(_sample_scorecard(), "test-analysis-id"))
    assert result is False


@patch("video_scorer.storage.supabase.settings")
def test_store_api_failure(mock_settings):
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = type("S", (), {"get_secret_value": lambda self: "test-key"})()

    mock_resp = AsyncMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("video_scorer.storage.supabase.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.patch = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(store_scorecard(_sample_scorecard(), "test-analysis-id"))
        assert result is False


@patch("video_scorer.storage.supabase.settings")
def test_store_network_error(mock_settings):
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = type("S", (), {"get_secret_value": lambda self: "test-key"})()

    with patch("video_scorer.storage.supabase.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.patch = AsyncMock(side_effect=httpx.ConnectError("network down"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(store_scorecard(_sample_scorecard(), "test-analysis-id"))
        assert result is False
