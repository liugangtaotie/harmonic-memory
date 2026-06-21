"""Qdrant vector store client — semantic search and embedding storage."""

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import config


class MemoryQdrant:
    """Wrapper around Qdrant for memory vector operations."""

    def __init__(self):
        self.client = QdrantClient(url=config.storage.qdrant.url)
        self.collection = config.storage.qdrant.collection
        self.vector_size = config.storage.qdrant.vector_size

    def ensure_collection(self):
        """Ensure the memory collection exists with correct schema."""
        try:
            info = self.client.get_collection(self.collection)
            # Check if vector size matches, recreate if not
            if info.config.params.vectors.size != self.vector_size:
                self.client.delete_collection(self.collection)
                raise UnexpectedResponse(
                    status_code=404, reason="vector size mismatch"
                )
        except (UnexpectedResponse, Exception):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def get_collection_info(self) -> dict:
        """Get collection health and stats."""
        try:
            info = self.client.get_collection(self.collection)
            pts = getattr(info, 'points_count', 0)
            vecs = getattr(info, 'vectors_count', pts)  # some versions use points_count
            return {
                "name": self.collection,
                "status": str(info.status) if hasattr(info, 'status') else 'ok',
                "vectors_count": int(vecs),
                "segments_count": getattr(info, 'segments_count', 0),
            }
        except Exception as e:
            return {"error": str(e)}

    async def upsert(self, point_id: str, vector: list[float],
                     payload: dict | None = None) -> str:
        """Insert or update a vector point. Returns the point ID."""
        if payload is None:
            payload = {}
        payload["_id"] = point_id

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    async def search(
        self,
        vector: list[float],
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search for similar vectors. Returns list of {id, score, payload}."""
        try:
            results: list[ScoredPoint] = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload or {},
                }
                for r in results
            ]
        except Exception:
            return []

    async def search_batch(
        self,
        vectors: list[list[float]],
        limit: int = 5,
    ) -> list[list[dict]]:
        """Batch search for multiple query vectors."""
        try:
            results = self.client.search_batch(
                collection_name=self.collection,
                requests=[
                    {
                        "vector": v,
                        "limit": limit,
                    }
                    for v in vectors
                ],
            )
            return [
                [
                    {"id": r.id, "score": r.score, "payload": r.payload or {}}
                    for r in batch
                ]
                for batch in results
            ]
        except Exception:
            return [[] for _ in vectors]

    def delete(self, point_id: str):
        """Delete a vector point."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=[point_id],
            )
        except Exception:
            pass

    def count(self) -> int:
        """Get total vector count."""
        try:
            info = self.client.get_collection(self.collection)
            return getattr(info, 'points_count', 0) or getattr(info, 'vectors_count', 0) or 0
        except Exception:
            return 0

    def health(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False
