"""Memory decay — content-weighted exponential decay with archival."""

import math
import os
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import config
from ..db.sqlite import MemoryDB

logger = logging.getLogger(__name__)


def decay_score(
    created_at: str,
    memory_type: str = "fact",
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed_at: str | None = None,
) -> float:
    """Calculate memory decay score.

    Score = base_decay * importance_mod * access_mod * type_mod

    Returns 0.0 (fully decayed) to 1.0 (fresh).
    """
    cfg = config.lifecycle.decay

    # Days since creation
    try:
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_dt = datetime.now(timezone.utc)
    days_old = (datetime.now(timezone.utc) - created_dt).days

    # Base exponential decay
    decay_rate = math.log(2) / cfg.half_life_days
    base = math.exp(-decay_rate * days_old)

    # Importance modifier (important memories decay slower)
    importance_mod = 0.5 + (importance * 0.5)

    # Access frequency modifier
    if access_count > 0:
        access_mod = min(2.0, 1.0 + (access_count * 0.1))
    else:
        access_mod = 1.0

    # Type modifier
    type_mod = cfg.type_weights.get(memory_type, 1.0)

    score = base * importance_mod * access_mod

    # Apply type modifier (0 = never decays, 2.0 = decays fast)
    if type_mod == 0.0:
        return 1.0  # Never decay
    score = score * (2.0 - type_mod)  # Invert: 0.3 = very slow decay

    return min(1.0, max(0.0, score))


async def run_decay_cycle(db: MemoryDB | None = None):
    """Run a full decay cycle: score all active memories, archive/decay as needed.

    This replaces the old decay.py script.
    """
    if db is None:
        db = MemoryDB()
        db.init_schema()

    cfg = config.lifecycle.decay
    archive_path = config.storage.archive.expanded_path

    # Get all active/decayed memories
    all_memories = []
    for state in ["active", "extracted", "decayed"]:
        all_memories.extend(db.list_memories(state=state, limit=10000))

    logger.info(f"Running decay cycle on {len(all_memories)} memories")

    archived = 0
    decayed = 0
    active_count = 0

    for mem in all_memories:
        score = decay_score(
            created_at=mem["created_at"],
            memory_type=mem["type"],
            importance=mem.get("importance", 0.5),
            access_count=mem.get("access_count", 0),
            last_accessed_at=mem.get("last_accessed_at"),
        )

        # Update score in DB
        db.update_memory(mem["id"], {"score": score})

        if score < cfg.archive_threshold:
            # Archive the memory
            new_state = "archived"
            db.update_memory(mem["id"], {"state": new_state})

            # Move to archive directory
            month_dir = archive_path / datetime.now().strftime("%Y-%m")
            month_dir.mkdir(parents=True, exist_ok=True)

            # Write memory as markdown for archival
            archive_file = month_dir / f"{mem['id']}.md"
            with open(archive_file, "w", encoding="utf-8") as f:
                f.write(f"---\nname: {mem['id']}\ntype: {mem['type']}\n")
                f.write(f"source: {mem['source']}\nstate: archived\n")
                f.write(f"score: {score:.3f}\ncreated: {mem['created_at']}\n---\n\n")
                f.write(f"# {mem.get('summary', mem['content'][:80])}\n\n")
                f.write(mem["content"])

            archived += 1
        elif score < cfg.decayed_threshold:
            db.update_memory(mem["id"], {"state": "decayed"})
            decayed += 1
        else:
            active_count += 1

    logger.info(
        f"Decay cycle complete: {active_count} active, "
        f"{decayed} decayed, {archived} archived"
    )

    # ── Hard storage limits ──
    storage_cfg = config.ingestion.limits
    active_total = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE state IN ('active','extracted')"
    ).fetchone()[0]

    if active_total > getattr(storage_cfg, "aggressive_decay_threshold", 1500):
        # Aggressive decay: archive lowest-score active memories until under limit
        excess = active_total - getattr(storage_cfg, "max_active_memories", 2000)
        if excess > 0:
            rows = db.conn.execute(
                "SELECT id, score FROM memories WHERE state IN ('active','extracted') "
                "ORDER BY score ASC, created_at ASC LIMIT ?",
                (min(excess, 500),),
            ).fetchall()
            for r in rows:
                db.update_memory(r["id"], {"state": "archived", "score": r["score"]})
                archived += 1
            logger.info(f"Aggressive decay: archived {len(rows)} memories (over limit)")

    return {
        "active": active_count,
        "decayed": decayed,
        "archived": archived,
        "total": len(all_memories),
        "hard_limit_enforced": active_total > getattr(storage_cfg, "aggressive_decay_threshold", 1500),
    }
