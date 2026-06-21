"""Codex watcher — poll Codex SQLite for new sessions."""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CODEX_DIR = Path(os.path.expanduser("~/.codex"))
LAST_SYNC_FILE = Path(os.path.expanduser("~/.harmonic-memory/codex_last_sync.json"))


def get_last_sync_time() -> str:
    """Get the timestamp of the last successful sync."""
    if LAST_SYNC_FILE.exists():
        try:
            data = json.loads(LAST_SYNC_FILE.read_text())
            return data.get("last_sync", "1970-01-01T00:00:00")
        except Exception:
            pass
    return "1970-01-01T00:00:00"


def save_last_sync_time(ts: str):
    """Save the last sync timestamp."""
    LAST_SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_SYNC_FILE.write_text(json.dumps({"last_sync": ts}))


def extract_codex_sessions(since_ts: str | None = None, max_sessions: int = 10) -> list[str]:
    """Extract text from recent Codex sessions.

    Args:
        since_ts: Only process sessions modified after this ISO timestamp
        max_sessions: Maximum number of sessions to process

    Returns:
        List of session texts
    """
    if since_ts is None:
        since_ts = get_last_sync_time()

    texts = []
    sessions_dir = CODEX_DIR / "sessions"

    if not sessions_dir.exists():
        return texts

    try:
        files = sorted(
            sessions_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[:max_sessions]:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                texts.append(content[:50000])
            except Exception as e:
                logger.warning(f"Error reading Codex session {f}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Error scanning Codex sessions: {e}")

    return texts


async def run_codex_sync(api_url: str = "http://127.0.0.1:18900"):
    """Run a Codex sync cycle — poll for new sessions and ingest."""
    import httpx

    since_ts = get_last_sync_time()
    new_ts = since_ts

    texts = extract_codex_sessions(since_ts=since_ts)

    if not texts:
        return {"status": "no_new_sessions"}

    ingested = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            try:
                resp = await client.post(
                    f"{api_url}/api/v1/ingest",
                    json={
                        "text": text,
                        "source": "codex",
                        "source_ref": "codex_sync",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ingested += data.get("memories_created", 0)
            except Exception as e:
                logger.warning(f"Codex sync ingestion failed: {e}")

    # Update sync timestamp
    new_ts = datetime.now(timezone.utc).isoformat()
    save_last_sync_time(new_ts)

    return {"status": "success", "sessions_processed": len(texts), "memories_created": ingested}
