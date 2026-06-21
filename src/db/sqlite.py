"""SQLite database — source of truth for all structured memory data."""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import config

# Allowed columns for dynamic INSERT
ALLOWED_COLUMNS = {
    "id", "type", "content", "summary", "source", "source_ref",
    "confidence", "importance", "state", "score", "embedding_id", "parent_id",
    "created_at", "updated_at", "last_accessed_at", "access_count", "metadata",
}

SCHEMA_SQL = """
-- Memories: the core table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN (
        'preference','fact','concept','procedure',
        'decision','question','event','relationship'
    )),
    content TEXT NOT NULL,
    summary TEXT,
    source TEXT NOT NULL DEFAULT 'manual',
    source_ref TEXT,
    confidence REAL DEFAULT 0.5,
    importance REAL DEFAULT 0.5,
    state TEXT DEFAULT 'extracted' CHECK(state IN (
        'raw_event','extracted','active','consolidated',
        'archived','rejected','decayed','forgotten'
    )),
    score REAL DEFAULT 1.0,
    embedding_id TEXT,
    parent_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

-- User profile: inferred attributes over time
CREATE TABLE IF NOT EXISTS user_profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_memory_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Memory relationships: edges in the knowledge graph
CREATE TABLE IF NOT EXISTS memory_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK(relation_type IN (
        'supports','contradicts','extends','replaces','references'
    )),
    weight REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);

-- Ingestion audit log
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_ref TEXT,
    raw_bytes INTEGER,
    extracted_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

-- Consolidated memory groups
CREATE TABLE IF NOT EXISTS consolidation_groups (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    member_ids TEXT NOT NULL DEFAULT '[]',
    consolidated_text TEXT,
    consolidated_memory_id TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_score ON memories(score);
CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_memory_edges_target ON memory_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_source ON ingestion_log(source);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_created ON ingestion_log(created_at);
CREATE INDEX IF NOT EXISTS idx_user_profile_category ON user_profile(category);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, summary,
    content='memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary)
    VALUES (new.rowid, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.rowid, old.content, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.rowid, old.content, old.summary);
    INSERT INTO memories_fts(rowid, content, summary)
    VALUES (new.rowid, new.content, new.summary);
END;
"""


class MemoryDB:
    """SQLite database manager for Harmonic Memory."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(config.storage.sqlite.expanded_path)
        self.db_path = Path(os.path.expanduser(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self):
        """Initialize database schema."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ─── Memory CRUD ───

    def insert_memory(self, memory: dict) -> str:
        """Insert a memory, returns its ID.

        Only includes fields that are present in the memory dict.
        Missing optional fields get DB defaults.
        """
        if "id" not in memory:
            memory["id"] = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Ensure required fields
        memory.setdefault("type", "fact")
        memory.setdefault("content", "")
        memory.setdefault("source", "manual")
        memory.setdefault("created_at", now)
        memory.setdefault("updated_at", now)

        # Set defaults for optional fields that are present in schema
        for opt_field, default in [
            ("summary", None), ("source_ref", None), ("confidence", 0.5),
            ("importance", 0.5), ("state", "extracted"), ("score", 1.0),
            ("embedding_id", None), ("parent_id", None),
            ("last_accessed_at", None), ("access_count", 0), ("metadata", "{}"),
        ]:
            memory.setdefault(opt_field, default)

        if isinstance(memory.get("metadata"), dict):
            memory["metadata"] = json.dumps(memory["metadata"], ensure_ascii=False)

        # Build dynamic INSERT with only present fields
        fields = [k for k in memory if k in ALLOWED_COLUMNS]
        placeholders = [f":{k}" for k in fields]
        sql = f"INSERT INTO memories ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
        params = {k: memory[k] for k in fields}

        self.conn.execute(sql, params)
        self.conn.commit()
        return memory["id"]

    def get_memory(self, memory_id: str) -> dict | None:
        """Get a memory by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def update_memory(self, memory_id: str, updates: dict) -> bool:
        """Update memory fields. Returns True if found."""
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"], ensure_ascii=False)

        set_clause = ", ".join(f"{k}=:{k}" for k in updates)
        updates["id"] = memory_id

        cursor = self.conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id=:id", updates
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def record_access(self, memory_id: str):
        """Record that a memory was accessed (for decay calculation)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE memories SET last_accessed_at=:now,
               access_count=access_count+1 WHERE id=:id""",
            {"now": now, "id": memory_id},
        )
        self.conn.commit()

    def delete_memory(self, memory_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def list_memories(
        self,
        type: str | None = None,
        state: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List memories with optional filters."""
        conditions = []
        params = {}
        if type:
            conditions.append("type=:type")
            params["type"] = type
        if state:
            conditions.append("state=:state")
            params["state"] = state
        if source:
            conditions.append("source=:source")
            params["source"] = source

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params["limit"] = limit
        params["offset"] = offset

        rows = self.conn.execute(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC "
            f"LIMIT :limit OFFSET :offset",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_keyword(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text keyword search via FTS5."""
        # Sanitize query for FTS5
        safe_query = " ".join(
            f'"{term}"' for term in query.split() if len(term) > 1
        )
        if not safe_query:
            return []

        rows = self.conn.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH :query
               ORDER BY rank
               LIMIT :limit""",
            {"query": safe_query, "limit": limit},
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ─── User Profile ───

    def get_profile(self, key: str | None = None, category: str | None = None) -> list[dict]:
        """Get user profile entries."""
        conditions = []
        params = {}
        if key:
            conditions.append("key=:key")
            params["key"] = key
        if category:
            conditions.append("category=:category")
            params["category"] = category
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = self.conn.execute(
            f"SELECT * FROM user_profile {where} ORDER BY confidence DESC", params
        ).fetchall()
        return [dict(r) for r in rows]

    def set_profile(self, key: str, value: str, category: str = "general",
                    confidence: float = 0.5, source_memory_ids: list[str] | None = None):
        """Upsert a user profile entry."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self.conn.execute(
            "SELECT * FROM user_profile WHERE key=?", (key,)
        ).fetchone()

        if existing:
            new_evidence = existing["evidence_count"] + 1
            new_confidence = (existing["confidence"] * existing["evidence_count"] + confidence) / new_evidence
            ids = json.loads(existing["source_memory_ids"])
            if source_memory_ids:
                ids.extend(source_memory_ids)
                ids = list(set(ids))

            self.conn.execute(
                """UPDATE user_profile SET value=:value, confidence=:confidence,
                   evidence_count=:evidence, source_memory_ids=:ids,
                   updated_at=:now, category=:category
                   WHERE key=:key""",
                {"value": value, "confidence": new_confidence, "evidence": new_evidence,
                 "ids": json.dumps(ids), "now": now, "category": category, "key": key},
            )
        else:
            self.conn.execute(
                """INSERT INTO user_profile (key, value, category, confidence,
                   evidence_count, source_memory_ids, created_at, updated_at)
                   VALUES (:key, :value, :category, :confidence, 1, :ids, :now, :now)""",
                {"key": key, "value": value, "category": category,
                 "confidence": confidence,
                 "ids": json.dumps(source_memory_ids or []), "now": now},
            )
        self.conn.commit()

    # ─── Memory Edges ───

    def add_edge(self, source_id: str, target_id: str,
                 relation_type: str = "references", weight: float = 1.0) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """INSERT INTO memory_edges (source_id, target_id, relation_type, weight, created_at)
               VALUES (:src, :tgt, :rel, :w, :now)""",
            {"src": source_id, "tgt": target_id, "rel": relation_type, "w": weight, "now": now},
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_edges(self, memory_id: str, direction: str = "both") -> list[dict]:
        """Get edges for a memory. direction: 'out', 'in', or 'both'."""
        if direction == "out":
            rows = self.conn.execute(
                "SELECT * FROM memory_edges WHERE source_id=?", (memory_id,)
            ).fetchall()
        elif direction == "in":
            rows = self.conn.execute(
                "SELECT * FROM memory_edges WHERE target_id=?", (memory_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_edges WHERE source_id=? OR target_id=?",
                (memory_id, memory_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Ingestion Log ───

    def log_ingestion(self, **kwargs) -> int:
        kwargs.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        kwargs.setdefault("status", "pending")
        kwargs.setdefault("metadata", "{}")
        if isinstance(kwargs.get("metadata"), dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"], ensure_ascii=False)

        fields = ", ".join(kwargs.keys())
        placeholders = ", ".join(f":{k}" for k in kwargs)
        cursor = self.conn.execute(
            f"INSERT INTO ingestion_log ({fields}) VALUES ({placeholders})", kwargs
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_ingestion(self, log_id: int, status: str, error: str | None = None,
                         extracted_count: int | None = None):
        updates = {"status": status, "id": log_id}
        extras = []
        if error:
            updates["error_message"] = error
        if extracted_count is not None:
            updates["extracted_count"] = extracted_count
        set_clause = ", ".join(f"{k}=:{k}" for k in updates if k != "id")
        self.conn.execute(
            f"UPDATE ingestion_log SET {set_clause} WHERE id=:id", updates
        )
        self.conn.commit()

    def get_ingestion_stats(self, hours: int = 24) -> dict:
        """Get ingestion statistics."""
        row = self.conn.execute(
            """SELECT COUNT(*) as total, SUM(extracted_count) as total_extracted,
               AVG(latency_ms) as avg_latency
               FROM ingestion_log
               WHERE created_at > datetime('now', '-' || ? || ' hours')""",
            (hours,),
        ).fetchone()
        return dict(row)

    # ─── Stats ───

    def get_stats(self) -> dict:
        """Get overall database statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM memories GROUP BY type"
        ).fetchall():
            by_type[row["type"]] = row["cnt"]

        by_state = {}
        for row in self.conn.execute(
            "SELECT state, COUNT(*) as cnt FROM memories GROUP BY state"
        ).fetchall():
            by_state[row["state"]] = row["cnt"]

        edges = self.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
        profile = self.conn.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0]

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_state": by_state,
            "total_edges": edges,
            "profile_attributes": profile,
        }

    # ─── Helpers ───

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse JSON fields
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except json.JSONDecodeError:
                d["metadata"] = {}
        return d

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
