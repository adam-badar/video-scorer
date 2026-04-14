"""Generate content-writer voice example markdown from three-way comparison deltas."""

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def sanitize_topic_slug(topic: str) -> str:
    """Sanitize topic string into a safe filename slug.

    Only allows [a-z0-9-], max 50 chars. Prevents path traversal.
    """
    slug = topic.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    return slug[:50] or "untitled"


def generate_voice_example_markdown(
    platform: str,
    ai_draft_text: str | None,
    final_script_text: str,
    spoken_transcript: str,
    delta_analysis: dict,
    topic: str | None = None,
) -> tuple[str, str]:
    """Generate voice example markdown content and suggested filename.

    Returns (markdown_content, suggested_filename).
    The caller is responsible for writing to the correct path.
    """
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    date_str = now_et.strftime("%Y-%m-%d")

    topic_slug = sanitize_topic_slug(topic or "comparison")
    filename = f"{date_str}-{topic_slug}.md"

    # Build markdown
    lines = [
        f"## {platform.capitalize()} - {date_str}",
        f"**Context:** Three-way comparison — AI draft vs final script vs spoken transcript",
        "",
    ]

    if ai_draft_text and ai_draft_text.strip():
        lines.append("**AI draft:**")
        lines.append(f"> {ai_draft_text.strip()[:500]}")
        lines.append("")

    lines.append("**Final script:**")
    lines.append(f"> {final_script_text.strip()[:500]}")
    lines.append("")

    lines.append("**Spoken transcript:**")
    lines.append(f"> {spoken_transcript.strip()[:500]}")
    lines.append("")

    # Deltas
    deltas = delta_analysis.get("deltas", [])
    if deltas:
        lines.append("**Draft vs actual notes:**")
        for d in deltas[:10]:
            what = d.get("what_changed", "")
            why = d.get("why", "")
            pattern = d.get("pattern_to_carry_forward", "")
            lines.append(f"- **{what}** — {why}")
            if pattern:
                lines.append(f"  - Pattern: {pattern}")
        lines.append("")

    # Voice patterns
    patterns = delta_analysis.get("voice_patterns", [])
    if patterns:
        lines.append("**Patterns to carry forward:**")
        for p in patterns:
            lines.append(f"- {p}")
        lines.append("")

    # Overall assessment
    overall = delta_analysis.get("overall_assessment", "")
    if overall:
        lines.append(f"**Overall:** {overall}")
        lines.append("")

    return "\n".join(lines), filename
