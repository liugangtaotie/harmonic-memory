"""Claude Code hook integration — process session transcripts."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_session_text(transcript_path: str | Path, max_exchanges: int = 30,
                         max_chars: int = 50000) -> str:
    """Extract the last N exchanges from a Claude Code transcript as text.

    Args:
        transcript_path: Path to the history.jsonl file
        max_exchanges: Maximum number of exchanges to include
        max_chars: Maximum total characters

    Returns:
        Concatenated text of user/assistant exchanges
    """
    path = Path(transcript_path)
    if not path.exists():
        return ""

    exchanges = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for line in lines[-200:]:  # Look at last 200 lines
                try:
                    entry = json.loads(line)
                    role = entry.get("role", "") or entry.get("type", "")
                    content = entry.get("content", "")

                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        )

                    if role in ("user", "assistant") and content.strip():
                        exchanges.append(f"{role}: {content[:1500]}")

                except (json.JSONDecodeError, KeyError):
                    continue

    except Exception as e:
        logger.warning(f"Error reading transcript {path}: {e}")
        return ""

    text = "\n".join(exchanges[-max_exchanges:])
    return text[:max_chars]


async def process_claude_transcript(
    transcript_path: str,
    api_url: str = "http://127.0.0.1:18900",
) -> dict[str, Any]:
    """Process a Claude Code transcript through the memory ingestion pipeline.

    Args:
        transcript_path: Path to history.jsonl
        api_url: Harmonic Memory API URL

    Returns:
        Ingestion result dict
    """
    import httpx

    text = extract_session_text(transcript_path)
    if not text or len(text) < 20:
        return {"status": "skipped", "reason": "insufficient_content"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{api_url}/api/v1/ingest",
            json={
                "text": text,
                "source": "claude",
                "source_ref": str(transcript_path),
            },
        )
        resp.raise_for_status()
        return resp.json()
