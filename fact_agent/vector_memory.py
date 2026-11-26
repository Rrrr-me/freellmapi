"""Vector-based memory retrieval using LLM API embeddings with local fallback."""

import hashlib
import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional

from fact_agent.config import get_settings
from fact_agent.http_client import LLMClient
from fact_agent.memory_retriever import keyword_retrieval
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)

# Embedding dimensions for common models
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "auto": 1536,  # Default assumption
}

VECTOR_DB_NAME = "vector_memory.db"
LOCAL_EMBEDDING_DIM = 256  # Local hash-based embedding dimension


def _serialize_vector(vector: list[float]) -> bytes:
    """Serialize vector to binary blob (little-endian float32)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _deserialize_vector(blob: bytes, dim: int) -> list[float]:
    """Deserialize binary blob to vector."""
    return list(struct.unpack(f"<{dim}f", blob))


def _local_embedding(text: str, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic local embedding using hashing.

    This provides semantic-ish similarity without external API calls.
    Uses multiple hash seeds to create a distributed representation.
    """
    # Use multiple hash functions for better distribution
    embedding = [0.0] * dim
    words = text.lower().split()

    for i, word in enumerate(words):
        # Create multiple hash values per word
        for seed in range(4):
            h = hashlib.md5(f"{word}:{seed}".encode()).digest()
            # Map to dimensions
            for j in range(0, len(h), 2):
                idx = (h[j] + (h[j+1] << 8)) % dim
                # Add weighted contribution
                embedding[idx] += 1.0 / (1.0 + i * 0.1)

    # Normalize
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding


class VectorMemory:
    """Vector memory store using SQLite + LLM API embeddings with local fallback."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        embedding_model: str = "auto",
        dimensions: Optional[int] = None,
        use_local_fallback: bool = True,
    ):
        self.settings = get_settings()
        self.db_path = db_path or (self.settings.memory_db_path.parent / VECTOR_DB_NAME)
        self.client = client
        self.embedding_model = embedding_model
        self.api_dimensions = dimensions or EMBEDDING_DIMENSIONS.get(embedding_model, 1536)
        self.use_local_fallback = use_local_fallback
        self._api_failed = False
        self._init_db()

    def _init_db(self) -> None:
        """Initialize vector memory database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS vector_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_type TEXT NOT NULL DEFAULT 'local',  -- 'api' or 'local'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)

            # Migration: add embedding_type column if missing
            cursor = db.execute("PRAGMA table_info(vector_memories)")
            cols = {row[1] for row in cursor.fetchall()}
            if "embedding_type" not in cols:
                db.execute("ALTER TABLE vector_memories ADD COLUMN embedding_type TEXT DEFAULT 'local'")
                db.execute("UPDATE vector_memories SET embedding_type = 'local' WHERE embedding_type IS NULL")

            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_category
                ON vector_memories(category)
            """)
            db.commit()

        logger.debug("Vector memory DB initialized", extra={"path": str(self.db_path)})

    def _get_embedding(self, text: str) -> tuple[list[float], str]:
        """Get embedding from LLM API or local fallback.

        Returns: (embedding_vector, embedding_type)
        """
        # Try API first if not already failed
        if self.client and not self._api_failed:
            try:
                return self._get_api_embedding(text)
            except Exception as e:
                logger.debug("Embedding API failed, switching to local", exc_info=e)
                self._api_failed = True

        # Local fallback
        if self.use_local_fallback:
            return _local_embedding(text), "local"

        # No fallback available
        return [0.0] * self.api_dimensions, "zero"

    def _get_api_embedding(self, text: str) -> tuple[list[float], str]:
        """Get embedding from LLM API (OpenAI-compatible embeddings endpoint)."""
        import urllib.request

        payload = {
            "model": self.embedding_model if self.embedding_model != "auto" else "text-embedding-3-small",
            "input": text,
        }
        headers = {"Authorization": f"Bearer {self.client.api_key}", "Content-Type": "application/json"}

        req = urllib.request.Request(
            self.client.api_url.replace("/chat/completions", "/embeddings"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        embedding = result["data"][0]["embedding"]
        return embedding, "api"

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def add_memory(self, memory_id: int, category: str, fact: str) -> bool:
        """Add or update a memory's vector embedding."""
        try:
            text = f"{category}: {fact}"
            embedding, emb_type = self._get_embedding(text)
            blob = _serialize_vector(embedding)

            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    INSERT OR REPLACE INTO vector_memories (memory_id, category, fact, embedding, embedding_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (memory_id, category, fact, blob, emb_type))
                db.commit()

            logger.debug("Added vector memory", extra={"memory_id": memory_id, "type": emb_type})
            return True

        except Exception as e:
            logger.error("Failed to add vector memory", exc_info=e)
            return False

    def search(
        self,
        query: str,
        limit: int = 8,
        min_similarity: float = 0.25,  # Higher threshold for local embeddings to avoid false positives
    ) -> list[dict[str, Any]]:
        """Search memories by vector similarity."""
        if not query.strip():
            return []

        try:
            query_embedding, _ = self._get_embedding(query)
            query_dim = len(query_embedding)

            with sqlite3.connect(self.db_path) as db:
                rows = db.execute("""
                    SELECT memory_id, category, fact, embedding
                    FROM vector_memories
                """).fetchall()

            if not rows:
                return []

            # Score all memories
            scored = []
            for memory_id, category, fact, blob in rows:
                try:
                    embedding = _deserialize_vector(blob, query_dim)
                    score = self._cosine_similarity(query_embedding, embedding)
                    if score >= min_similarity:
                        scored.append((score, memory_id, category, fact))
                except Exception:
                    continue  # Skip corrupted vectors

            scored.sort(key=lambda x: x[0], reverse=True)

            return [
                {"score": score, "id": mid, "category": cat, "fact": fact}
                for score, mid, cat, fact in scored[:limit]
            ]

        except Exception as e:
            logger.error("Vector search failed", exc_info=e)
            return []

    def rebuild_from_long_term(self, long_term_db: Optional[Path] = None) -> int:
        """Rebuild vector index from long-term memory database."""
        lt_db = long_term_db or self.settings.long_term_memory_db_path
        count = 0

        try:
            with sqlite3.connect(lt_db) as db:
                rows = db.execute("SELECT id, category, fact FROM memories").fetchall()

            for memory_id, category, fact in rows:
                if self.add_memory(memory_id, category, fact):
                    count += 1

            logger.info("Rebuilt vector index", extra={"count": count})
            return count

        except Exception as e:
            logger.error("Failed to rebuild vector index", exc_info=e)
            return 0


def retrieve_vector_memories(
    query: str,
    limit: int = 8,
    client: Optional[LLMClient] = None,
    long_term_db: Optional[Path] = None,
    vector_db: Optional[Path] = None,
) -> str:
    """Retrieve memories using vector search with keyword fallback."""
    settings = get_settings()

    # Create vector memory instance
    vm = VectorMemory(
        db_path=vector_db,
        client=client,
        embedding_model=settings.llm_model,
    )

    # Try vector search
    results = vm.search(query, limit=limit)

    if results:
        logger.debug("Vector search returned results", extra={"count": len(results)})
        return "\n".join(
            f"- [{r['category']}] {r['fact']} (similarity: {r['score']:.3f})"
            for r in results
        )

    # Fallback to keyword retrieval from long-term memory
    logger.debug("Vector search empty, falling back to keyword")

    lt_db = long_term_db or settings.long_term_memory_db_path
    try:
        with sqlite3.connect(lt_db) as db:
            rows = db.execute("SELECT id, category, fact FROM memories ORDER BY id DESC").fetchall()
    except Exception as e:
        logger.error("Failed to load memories for keyword fallback", exc_info=e)
        return ""

    return keyword_retrieval(rows, query, limit)