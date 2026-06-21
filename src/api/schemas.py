"""Pydantic models for the Harmonic Memory API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw text to extract memories from", min_length=10)
    source: str = Field(default="manual", description="Source: claude, codex, hermes, manual")
    source_ref: str | None = Field(default=None, description="Session ID, file path, etc.")
    use_fallback: bool = Field(default=False, description="Force DeepSeek fallback")


class IngestResponse(BaseModel):
    status: str
    memories_created: int = 0
    duplicates_skipped: int = 0
    rejected_low_quality: int = 0
    memory_ids: list[str] = []
    latency_ms: int = 0


class MemoryResponse(BaseModel):
    id: str
    type: str
    content: str
    summary: str | None = None
    source: str
    source_ref: str | None = None
    confidence: float
    importance: float
    state: str
    score: float
    created_at: str
    updated_at: str
    access_count: int
    metadata: dict[str, Any] | None = None


class MemoryUpdate(BaseModel):
    type: str | None = None
    content: str | None = None
    importance: float | None = None
    state: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    type: str | None = None
    source: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[MemoryResponse]
    total: int
    latency_ms: int


class ProfileEntry(BaseModel):
    key: str
    value: str
    category: str
    confidence: float
    evidence_count: int


class ProfileUpdate(BaseModel):
    key: str
    value: str
    category: str = "general"
    confidence: float = 0.5


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant: dict[str, Any]
    sqlite: dict[str, Any]
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_memories: int
    by_type: dict[str, int]
    by_state: dict[str, int]
    total_edges: int
    profile_attributes: int
    qdrant_vectors: int
    ingestion_24h: dict[str, Any]


# ─── Unified Search ───

class UnifiedSearchHit(BaseModel):
    source_type: str = "unknown"
    score: float = 0.0
    preview: str = ""
    file: str | None = None
    filename: str | None = None
    timestamp: str | None = None
    source_name: str | None = None
    source_table: str | None = None
    row_data: dict[str, Any] | None = None


class UnifiedSearchResponse(BaseModel):
    query: str
    results: list[UnifiedSearchHit]
    total: int
    sources: dict[str, int]
    latency_ms: int
