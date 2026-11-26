"""Long-term memory fact extraction."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from fact_agent.config import get_settings
from fact_agent.http_client import LLMClient
from fact_agent.exceptions import MemoryExtractionError
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """Extract useful, lasting facts explicitly stated by the user.

Return ONLY valid JSON:
[
  {
    "category": "Project",
    "fact": "User is developing Fact Agent using Python."
  }
]

Return [] if there is nothing worth remembering.
Never invent facts."""


def extract_memories(user_message: str, client: LLMClient | None = None) -> list[dict[str, str]]:
    """Extract facts from user message using LLM."""
    settings = get_settings()
    client = client or LLMClient(
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        # Parse JSON array from response
        start = response.find("[")
        end = response.rfind("]")

        if start == -1 or end == -1:
            logger.debug("No JSON array found in extraction response")
            return []

        content = response[start:end + 1]
        memories = json.loads(content)

        if not isinstance(memories, list):
            logger.warning("Extraction response is not a list", extra={"type": type(memories).__name__})
            return []

        # Validate each memory
        validated = []
        for item in memories:
            if not isinstance(item, dict):
                continue

            category = item.get("category", "").strip()
            fact = item.get("fact", "").strip()

            if not category or not fact:
                continue
            if len(fact) > settings.max_fact_length:
                logger.warning("Fact too long, skipping", extra={"length": len(fact)})
                continue

            validated.append({"category": category, "fact": fact})

        return validated

    except Exception as e:
        logger.error("Memory extraction failed", exc_info=e)
        raise MemoryExtractionError(f"Extraction failed: {e}") from e


def store_memories(memories: list[dict[str, str]], db_path: str | None = None, client: LLMClient | None = None) -> int:
    """Store extracted memories in database, avoiding duplicates.

    Also adds vector embeddings for vector search.
    """
    settings = get_settings()
    db_path = db_path or str(settings.long_term_memory_db_path)

    if not memories:
        return 0

    stored = 0
    try:
        with sqlite3.connect(db_path) as db:
            for item in memories:
                category = item["category"]
                fact = item["fact"]

                cursor = db.execute(
                    """
                    INSERT INTO memories (category, fact)
                    SELECT ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memories WHERE fact = ?
                    )
                    """,
                    (category, fact, fact),
                )
                if cursor.rowcount > 0:
                    stored += 1
                    memory_id = cursor.lastrowid

                    # Add vector embedding for new memory
                    if client and memory_id:
                        try:
                            from fact_agent.vector_memory import VectorMemory
                            vm = VectorMemory(client=client)
                            vm.add_memory(memory_id, category, fact)
                        except Exception as e:
                            logger.debug("Failed to add vector embedding", exc_info=e)

        logger.info("Stored memories", extra={"count": stored, "total": len(memories)})
        return stored

    except sqlite3.Error as e:
        logger.error("Database error storing memories", exc_info=e)
        raise MemoryExtractionError(f"Database error: {e}") from e


def extract_and_store(user_message: str, client: LLMClient | None = None, db_path: str | None = None) -> int:
    """Extract facts from user message and store them."""
    if not get_settings().enable_memory_extraction:
        return 0

    memories = extract_memories(user_message, client)
    if not memories:
        return 0

    return store_memories(memories, db_path, client)