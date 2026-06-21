"""Real-time session watcher — scans OpenClaw + Claude Code transcripts,
extracts structured memories via LLM, stores in memories table.

Called via API endpoint or scheduled cron.
"""

import json
import os
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ingest import ingest as ingest_pipeline
from ..db.sqlite import MemoryDB

logger = logging.getLogger(__name__)


def _expand(p: str) -> str:
    return os.path.expanduser(p)


# Session transcript directories
SESSION_DIRS = [
    _expand(r"~\.openclaw\agents\main\sessions"),
    _expand(r"~\.claude\projects\--wsl-localhost-Ubuntu-24-04-tmp-skillopt-sleep-claude--h1o6amp"),
]

# Track which files have already been ingested (simple last-ingest-time based)
TRACKER_FILE = _expand(r"~\.harmonic-memory\ingest_tracker.json")


def _load_tracker() -> dict:
    """Load tracker: {filepath: last_ingested_mtime}."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_tracker(tracker: dict):
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f)


def _extract_conversation_text(filepath: str) -> str | None:
    """Extract conversational text from a .jsonl session file.

    OpenClaw format: lines are JSON with role/content.
    Claude Code format: lines are JSON with message/content.
    """
    messages = []
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = None
                role = None

                # OpenClaw session format
                if "role" in msg and "content" in msg:
                    role = msg["role"]
                    content = msg["content"]
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                # Claude Code format: message.content
                elif "message" in msg:
                    inner = msg["message"]
                    if isinstance(inner, dict):
                        role = inner.get("role", "unknown")
                        content = inner.get("content", "")
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = " ".join(
                                c.get("text", "") for c in content if isinstance(c, dict)
                            )

                if text and text.strip() and role in ("user", "assistant", "human", "ai"):
                    messages.append(f"[{role}] {text.strip()}")

    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None

    if not messages:
        return None

    # Join with separators, truncate to ~8000 chars for LLM extraction
    full_text = "\n\n".join(messages)
    if len(full_text) > 12000:
        # Keep first 4000 and last 8000 chars (most recent conversation is usually most relevant)
        full_text = full_text[:4000] + "\n\n...(省略中间)...\n\n" + full_text[-8000:]

    return full_text


async def ingest_recent_sessions(
    db: MemoryDB | None = None,
    max_files: int = 5,
    min_mtime_delta: int = 60,  # only ingest files at least 60s old (allow writes to finish)
) -> dict[str, Any]:
    """Scan recent session transcript files, extract memories, store.

    Args:
        db: MemoryDB instance
        max_files: Max new files to process per run
        min_mtime_delta: Minimum seconds since last modification (avoid in-flight writes)

    Returns:
        Summary with files_scanned, files_ingested, memories_created
    """
    if db is None:
        db = MemoryDB()
        db.init_schema()

    tracker = _load_tracker()
    now = time.time()
    files_ingested = 0
    total_memories = 0

    all_files = []
    for sdir in SESSION_DIRS:
        if not os.path.isdir(sdir):
            continue
        try:
            for entry in os.scandir(sdir):
                if entry.is_file() and entry.name.endswith(".jsonl"):
                    all_files.append(entry.path)
        except OSError:
            pass

    # Sort by mtime descending (newest first)
    all_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    processed = 0
    for filepath in all_files:
        if processed >= max_files:
            break

        mtime = os.path.getmtime(filepath)
        filekey = filepath.replace("\\", "/")

        # Skip if already ingested at this mtime
        prev = tracker.get(filekey)
        if isinstance(prev, dict) and prev.get("mtime") == mtime:
            continue
        if prev == mtime:  # legacy format
            continue

        # Skip if file was modified too recently (might still be writing)
        if now - mtime < min_mtime_delta:
            continue

        file_size = os.path.getsize(filepath)
        # Skip empty/small files
        if file_size < 500:
            tracker[filekey] = {"mtime": mtime, "size": file_size}
            continue

        # Also skip if file size hasn't changed significantly
        if isinstance(prev, dict) and prev.get("size", 0) == file_size:
            tracker[filekey] = {"mtime": mtime, "size": file_size}
            continue

        text = _extract_conversation_text(filepath)
        if not text or len(text.strip()) < 100:
            tracker[filekey] = {"mtime": mtime, "size": file_size}
            continue

        filename = os.path.basename(filepath)
        logger.info(f"Ingesting session: {filename} ({file_size // 1024}KB)")

        try:
            result = await ingest_pipeline(
                text=text,
                source="openclaw_session" if "openclaw" in filepath else "claude_session",
                source_ref=filename,
                db=db,
                use_fallback=False,
            )
            total_memories += result.get("memories_created", 0)
            files_ingested += 1
            tracker[filekey] = {"mtime": mtime, "size": file_size}
            processed += 1
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")

    _save_tracker(tracker)

    return {
        "status": "ok",
        "files_scanned": len(all_files),
        "files_ingested": files_ingested,
        "memories_created": total_memories,
        "total_memories_now": db.get_stats()["total_memories"],
    }
