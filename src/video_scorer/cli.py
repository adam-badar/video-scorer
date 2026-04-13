"""CLI entry point: video-score analyze <file> --platform tiktok"""

import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import typer

from video_scorer.analyzers import ffmpeg
from video_scorer.analyzers.deepgram import analyze_transcript, transcribe
from video_scorer.qualitative.gemini import analyze as gemini_analyze
from video_scorer.scoring.hooks import check_hooks
from video_scorer.scoring.scorecard import compute_scorecard
from video_scorer.storage.supabase import store_scorecard

app = typer.Typer(help="Video content optimization scorer")


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to the video file"),
    platform: str = typer.Option("tiktok", help="Target platform: tiktok, youtube, instagram, linkedin"),
    qualitative: bool = typer.Option(False, "--qualitative", help="Run Gemini qualitative analysis (requires API key)"),
    store: bool = typer.Option(False, "--store", help="Store scorecard in Supabase (requires credentials)"),
    output_format: str = typer.Option("json", "--format", help="Output format: json or summary"),
):
    """Analyze a video file and output a scored report."""
    # Validate file
    path = Path(file).resolve()
    if not path.is_file():
        typer.echo(f"Error: File not found: {path}", err=True)
        raise typer.Exit(1)

    if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        typer.echo(f"Error: Unsupported file type: {path.suffix}", err=True)
        raise typer.Exit(1)

    if platform not in {"tiktok", "youtube", "instagram", "linkedin"}:
        typer.echo(f"Error: Unknown platform: {platform}. Use tiktok, youtube, instagram, or linkedin.", err=True)
        raise typer.Exit(1)

    # Check FFmpeg
    try:
        ffmpeg.probe_video(path)
    except FileNotFoundError:
        typer.echo("Error: FFmpeg not found. Install via: brew install ffmpeg", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Analyzing: {path.name} (platform: {platform})", err=True)

    # Run pipeline
    scorecard = asyncio.run(_run_pipeline(path, platform, qualitative, store))

    # Output
    if output_format == "summary":
        _print_summary(scorecard)
    else:
        typer.echo(json.dumps(scorecard, indent=2, default=str))


async def _run_pipeline(path: Path, platform: str, qualitative: bool, store: bool) -> dict:
    """Run the full analysis pipeline."""
    # Step 1: Video metadata + file hash
    typer.echo("  [1/5] Probing video...", err=True)
    info = ffmpeg.probe_video(path)
    video_hash = ffmpeg.file_hash(path)
    duration = info["duration_seconds"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 2: FFmpeg local analyzers
        typer.echo("  [2/5] Running FFmpeg analyzers (cuts, loudness, silence, loop)...", err=True)
        cuts = ffmpeg.detect_cuts(path)
        loop_score = ffmpeg.compute_loop_score(path)

        wav_path = None
        loudness = {"loudness_lufs": None, "true_peak_dbtp": None}
        silence = {"silence_ratio": 0, "silence_gaps": [], "max_silence_gap_ms": 0}

        try:
            wav_path = ffmpeg.extract_audio(path, tmpdir)
            loudness = ffmpeg.measure_loudness(wav_path)
            silence = ffmpeg.detect_silence(wav_path)
        except RuntimeError as e:
            typer.echo(f"  Warning: Audio extraction failed: {e}", err=True)

        # Step 3: Deepgram transcription
        typer.echo("  [3/5] Transcribing with Deepgram...", err=True)
        deepgram_raw = None
        transcript_analysis = None
        if wav_path and wav_path.exists():
            deepgram_raw = await transcribe(str(wav_path))

        if deepgram_raw:
            transcript_analysis = analyze_transcript(deepgram_raw)
            typer.echo(f"  Transcript: {transcript_analysis.get('word_count', 0)} words, WPM: {transcript_analysis.get('wpm', 'N/A')}", err=True)
        else:
            from video_scorer.config import settings
            if not settings.deepgram_api_key:
                typer.echo("  Skipped: Set VIDEO_SCORER_DEEPGRAM_API_KEY for full analysis.", err=True)
            else:
                typer.echo("  Warning: Deepgram transcription failed. Using local-only metrics.", err=True)

        # Step 4: Hook checks
        typer.echo("  [4/5] Checking hooks...", err=True)
        words = []
        if transcript_analysis and transcript_analysis.get("language_warning") is None:
            # Only check hooks if we have English transcript
            words = []
            if deepgram_raw:
                channels = deepgram_raw.get("results", {}).get("channels", [{}])
                if channels:
                    alts = channels[0].get("alternatives", [{}])
                    if alts:
                        words = alts[0].get("words", [])
        hook_checks = check_hooks(words, duration)

        # Step 5: Scorecard
        typer.echo("  [5/5] Computing scorecard...", err=True)
        pacing = {
            "wpm": transcript_analysis["wpm"] if transcript_analysis else None,
            "filler_word_count": transcript_analysis["filler_word_count"] if transcript_analysis else None,
        }
        audio_metrics = {
            "loudness_lufs": loudness.get("loudness_lufs"),
            "true_peak_dbtp": loudness.get("true_peak_dbtp"),
            "silence_ratio": silence.get("silence_ratio", 0),
            "max_silence_gap_ms": silence.get("max_silence_gap_ms", 0),
        }

        score = compute_scorecard(
            pacing=pacing,
            editing=cuts,
            audio=audio_metrics,
            hook_checks=hook_checks,
            loop_score=loop_score,
            duration_seconds=duration,
            platform=platform,
        )

    # Build full output
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))

    scorecard = {
        "video": {
            "file": path.name,
            "file_hash": video_hash,
            "duration_seconds": round(duration, 1),
            "resolution": info["resolution"],
        },
        "pacing": {
            "wpm": transcript_analysis["wpm"] if transcript_analysis else None,
            "filler_word_count": transcript_analysis["filler_word_count"] if transcript_analysis else None,
            "filler_words": transcript_analysis["filler_words"] if transcript_analysis else None,
            "max_silence_gap_ms": transcript_analysis.get("max_silence_gap_ms") if transcript_analysis else None,
            "pace_variation_stddev": transcript_analysis.get("pace_variation_stddev") if transcript_analysis else None,
            "detected_language": transcript_analysis.get("detected_language") if transcript_analysis else None,
            "language_confidence": transcript_analysis.get("language_confidence") if transcript_analysis else None,
            "language_warning": transcript_analysis.get("language_warning") if transcript_analysis else None,
        },
        "editing": cuts,
        "audio": {
            **audio_metrics,
            "silence_gaps": silence.get("silence_gaps", []),
        },
        "structure": {
            "loop_score": loop_score,
            "hook_checks": hook_checks,
        },
        "sentiment": transcript_analysis.get("sentiment") if transcript_analysis else None,
        "topics": transcript_analysis.get("topics") if transcript_analysis else None,
        "score": score,
        "qualitative": None,
        "scored_at": now_et.isoformat(),
    }

    if qualitative:
        typer.echo("  [+] Running Gemini qualitative analysis...", err=True)
        transcript_text = transcript_analysis.get("transcript", "") if transcript_analysis else ""
        qual_result = await gemini_analyze(scorecard, transcript_text)
        if qual_result:
            scorecard["qualitative"] = qual_result
            typer.echo("  Gemini analysis complete.", err=True)

    if store:
        import uuid
        typer.echo("  [+] Storing scorecard in Supabase...", err=True)
        cli_analysis_id = str(uuid.uuid4())
        stored = await store_scorecard(scorecard, cli_analysis_id)
        if stored:
            typer.echo("  Stored successfully.", err=True)

    return scorecard


def _print_summary(scorecard: dict):
    """Print a human-readable summary."""
    score = scorecard["score"]
    video = scorecard["video"]
    pacing = scorecard["pacing"]

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  VIDEO SCORE: {score['grade']} ({score['total']}/{score['max_possible']} = {score['percentage']}%)")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  File:     {video['file']}")
    typer.echo(f"  Duration: {video['duration_seconds']}s")
    typer.echo(f"  Platform: {score['platform_targets']['platform']}")
    typer.echo()

    for cat, pts in score["breakdown"].items():
        typer.echo(f"  {cat.capitalize():12s} {pts} pts")
    typer.echo()

    if pacing.get("wpm") is not None:
        typer.echo(f"  WPM: {pacing['wpm']} (target: {score['platform_targets']['wpm_target']})")
    if pacing.get("filler_word_count") is not None:
        typer.echo(f"  Fillers: {pacing['filler_word_count']}")
    if pacing.get("language_warning"):
        typer.echo(f"  Warning: {pacing['language_warning']}")

    typer.echo(f"  Cuts/min: {scorecard['editing']['cuts_per_minute']} (target: {score['platform_targets']['cuts_target']})")
    typer.echo(f"  Loudness: {scorecard['audio'].get('loudness_lufs', 'N/A')} LUFS")
    typer.echo(f"  Loop:     {scorecard['structure']['loop_score']}")
    typer.echo()


if __name__ == "__main__":
    app()
