"""FFmpeg-based local video analyzers: cuts, loudness, silence, loop score, duration."""

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def probe_video(path: Path) -> dict:
    """Get video metadata via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(f"Not a valid video file: {path.name}\nffprobe: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0))

    width, height = None, None
    for stream in data.get("streams", []):
        if stream.get("width"):
            width = stream["width"]
            height = stream["height"]
            break

    resolution = f"{width}x{height}" if width and height else "unknown"
    return {"duration_seconds": duration, "resolution": resolution}


def extract_audio(path: Path, tmpdir: str) -> Path:
    """Extract audio as 16kHz mono WAV."""
    wav_path = Path(tmpdir) / "audio.wav"
    duration = probe_video(path)["duration_seconds"]
    timeout = max(30, int(duration * 2 + 30))

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path),
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr.strip()}")
    return wav_path


def detect_cuts(path: Path) -> dict:
    """Detect scene changes using FFmpeg scdet filter."""
    info = probe_video(path)
    duration = info["duration_seconds"]
    if duration < 1:
        return {"cuts_per_minute": 0.0, "cut_timestamps": [], "total_cuts": 0}

    timeout = max(30, int(duration * 2 + 30))
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-vf", "scdet=threshold=0.3:sc_pass=1",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=timeout,
    )

    timestamps = []
    for line in result.stderr.splitlines():
        match = re.search(r"lavfi\.scd\.time:\s*([\d.]+)", line)
        if match:
            timestamps.append(float(match.group(1)))

    total_cuts = len(timestamps)
    cuts_per_minute = (total_cuts / duration) * 60 if duration > 0 else 0.0

    return {
        "cuts_per_minute": round(cuts_per_minute, 1),
        "cut_timestamps": timestamps,
        "total_cuts": total_cuts,
    }


def measure_loudness(wav_path: Path) -> dict:
    """Measure integrated loudness and true peak via ebur128."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", "ebur128=peak=true",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=60,
    )

    loudness_lufs = None
    true_peak = None

    for line in result.stderr.splitlines():
        if "I:" in line and "LUFS" in line:
            match = re.search(r"I:\s*([-\d.]+)\s*LUFS", line)
            if match:
                loudness_lufs = float(match.group(1))
        if "Peak:" in line and "dBFS" in line:
            match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", line)
            if match:
                true_peak = float(match.group(1))

    return {
        "loudness_lufs": loudness_lufs,
        "true_peak_dbtp": true_peak,
    }


def detect_silence(wav_path: Path) -> dict:
    """Detect silence gaps using silencedetect filter."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", "silencedetect=noise=-30dB:d=0.3",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=60,
    )

    gaps = []
    start = None
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            match = re.search(r"silence_start:\s*([\d.]+)", line)
            if match:
                start = float(match.group(1))
        elif "silence_end:" in line and start is not None:
            match = re.search(r"silence_end:\s*([\d.]+)", line)
            if match:
                end = float(match.group(1))
                gaps.append({"start": round(start, 2), "end": round(end, 2)})
                start = None

    # Get audio duration for ratio
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(wav_path)],
        capture_output=True, text=True, timeout=10,
    )
    audio_duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 1))

    total_silence = sum(g["end"] - g["start"] for g in gaps)
    silence_ratio = round(total_silence / audio_duration, 3) if audio_duration > 0 else 0.0
    max_gap_ms = max((int((g["end"] - g["start"]) * 1000) for g in gaps), default=0)

    return {
        "silence_ratio": silence_ratio,
        "silence_gaps": gaps,
        "max_silence_gap_ms": max_gap_ms,
    }


def compute_loop_score(path: Path) -> float:
    """Cosine similarity between first and last frames."""
    info = probe_video(path)
    duration = info["duration_seconds"]
    if duration < 2:
        return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        first_path = Path(tmpdir) / "first.png"
        last_path = Path(tmpdir) / "last.png"

        # Extract first frame
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vf", "select=eq(n\\,0)", "-vframes", "1", str(first_path)],
            capture_output=True, timeout=15,
        )
        # Extract last frame
        last_ts = max(0, duration - 0.1)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(last_ts), "-i", str(path), "-vframes", "1", str(last_path)],
            capture_output=True, timeout=15,
        )

        if not first_path.exists() or not last_path.exists():
            return 0.0

        first_img = np.array(Image.open(first_path).resize((64, 64)).convert("RGB")).flatten().astype(float)
        last_img = np.array(Image.open(last_path).resize((64, 64)).convert("RGB")).flatten().astype(float)

        dot = np.dot(first_img, last_img)
        norm = np.linalg.norm(first_img) * np.linalg.norm(last_img)
        if norm == 0:
            return 0.0
        return round(float(dot / norm), 3)
