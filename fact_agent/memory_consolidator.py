"""Memory consolidation and maintenance for Fact Agent."""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fact_agent.config import get_settings
from fact_agent.http_client import LLMClient
from fact_agent.memory_extractor import extract_memories, store_memories
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)

CONSOLIDATION_PROMPT = """Summarize the following conversation into key facts and decisions.
Focus on:
- Explicit user preferences stated
- Project decisions made
- Important facts revealed
- Action items or commitments

Return ONLY valid JSON:
{
  "summary": "Brief 2-3 sentence summary",
  "key_facts": [
    {"category": "Preference", "fact": "User prefers dark mode"},
    {"category": "Project", "fact": "User decided to use PostgreSQL for the database"}
  ],
  "decisions": ["Decision 1", "Decision 2"]
}

If nothing worth remembering, return {"summary": "", "key_facts": [], "decisions": []}."""


@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""
    messages_processed: int
    summary: str
    key_facts_extracted: int
    decisions_recorded: int
    memories_stored: int


class MemoryConsolidator:
    """Handles conversation summarization and memory maintenance."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        conversation_db: Optional[Path] = None,
        long_term_db: Optional[Path] = None,
        vector_db: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.client = client
        self.conversation_db = conversation_db or self.settings.memory_db_path
        self.long_term_db = long_term_db or self.settings.long_term_memory_db_path
        self.vector_db = vector_db

    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self.client:
            return self.client
        return LLMClient(
            api_url=self.settings.llm_api_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout=self.settings.llm_timeout,
        )

    def get_unconsolidated_messages(self, limit: int = 50) -> list[dict]:
        """Get messages that haven't been consolidated yet."""
        with sqlite3.connect(self.conversation_db) as db:
            # Check if consolidation tracking column exists
            cursor = db.execute("PRAGMA table_info(conversations)")
            cols = {row[1] for row in cursor.fetchall()}

            if "consolidated" not in cols:
                # Add column if missing
                db.execute("ALTER TABLE conversations ADD COLUMN consolidated INTEGER DEFAULT 0")
                db.commit()

            rows = db.execute("""
                SELECT id, role, content, created_at
                FROM conversations
                WHERE consolidated = 0
                ORDER BY id ASC
                LIMIT ?
            """, (limit,)).fetchall()

        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]

    def mark_consolidated(self, message_ids: list[int]) -> None:
        """Mark messages as consolidated."""
        if not message_ids:
            return
        placeholders = ",".join("?" * len(message_ids))
        with sqlite3.connect(self.conversation_db) as db:
            db.execute(
                f"UPDATE conversations SET consolidated = 1 WHERE id IN ({placeholders})",
                message_ids,
            )
            db.commit()

    def consolidate(
        self,
        max_messages: int = 50,
        min_messages: int = 10,
    ) -> Optional[ConsolidationResult]:
        """Consolidate unprocessed messages into long-term memories."""
        client = self._get_llm_client()
        messages = self.get_unconsolidated_messages(max_messages)

        if len(messages) < min_messages:
            logger.debug("Not enough messages to consolidate", extra={"count": len(messages)})
            return None

        # Build conversation text
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        try:
            # Ask LLM to summarize
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": CONSOLIDATION_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            import json
            start = response.find("{")
            end = response.rfind("}")

            if start == -1 or end == -1:
                logger.warning("No JSON in consolidation response")
                return None

            result = json.loads(response[start:end + 1])

            summary = result.get("summary", "").strip()
            key_facts = result.get("key_facts", [])
            decisions = result.get("decisions", [])

            # Store key facts as memories
            stored = 0
            if key_facts:
                stored = store_memories(key_facts, str(self.long_term_db), client)

            # Store summary as a memory if present
            if summary:
                summary_fact = {"category": "Summary", "fact": summary}
                stored += store_memories([summary_fact], str(self.long_term_db), client)

            # Mark messages as consolidated
            message_ids = [m["id"] for m in messages]
            self.mark_consolidated(message_ids)

            result_obj = ConsolidationResult(
                messages_processed=len(messages),
                summary=summary,
                key_facts_extracted=len(key_facts),
                decisions_recorded=len(decisions),
                memories_stored=stored,
            )

            logger.info("Memory consolidation complete", extra=result_obj.__dict__)
            return result_obj

        except Exception as e:
            logger.error("Consolidation failed", exc_info=e)
            return None

    def auto_consolidate_if_needed(self, threshold: int = 30) -> Optional[ConsolidationResult]:
        """Run consolidation if enough unconsolidated messages exist."""
        messages = self.get_unconsolidated_messages(limit=100)
        if len(messages) >= threshold:
            return self.consolidate()
        return None


@dataclass
class TTLPolicy:
    """Time-to-live policy for memories."""
    # Days after which to expire (None = never)
    conversation_days: Optional[int] = 90
    long_term_days: Optional[int] = 365
    # Max memories to keep per category (None = unlimited)
    max_per_category: Optional[int] = 100


class MemoryJanitor:
    """Handles memory cleanup based on TTL policies."""

    def __init__(
        self,
        policy: Optional[TTLPolicy] = None,
        conversation_db: Optional[Path] = None,
        long_term_db: Optional[Path] = None,
    ):
        self.policy = policy or TTLPolicy()
        self.conversation_db = conversation_db or get_settings().memory_db_path
        self.long_term_db = long_term_db or get_settings().long_term_memory_db_path

    def cleanup_conversations(self) -> int:
        """Delete old conversation messages based on TTL."""
        if self.policy.conversation_days is None:
            return 0

        cutoff = time.time() - (self.policy.conversation_days * 86400)
        cutoff_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff))

        with sqlite3.connect(self.conversation_db) as db:
            cursor = db.execute(
                "DELETE FROM conversations WHERE created_at < ?",
                (cutoff_str,),
            )
            db.commit()
            deleted = cursor.rowcount

        if deleted:
            logger.info("Cleaned up old conversations", extra={"deleted": deleted})
        return deleted

    def cleanup_long_term(self) -> int:
        """Delete old long-term memories based on TTL and category limits."""
        if self.policy.long_term_days is None and self.policy.max_per_category is None:
            return 0

        deleted = 0

        with sqlite3.connect(self.long_term_db) as db:
            # TTL-based cleanup
            if self.policy.long_term_days is not None:
                cutoff = time.time() - (self.policy.long_term_days * 86400)
                cutoff_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff))
                cursor = db.execute(
                    "DELETE FROM memories WHERE created_at < ?",
                    (cutoff_str,),
                )
                deleted += cursor.rowcount

            # Per-category limit cleanup
            if self.policy.max_per_category is not None:
                # Get categories with too many entries
                cats = db.execute("""
                    SELECT category, COUNT(*) as cnt
                    FROM memories
                    GROUP BY category
                    HAVING cnt > ?
                """, (self.policy.max_per_category,)).fetchall()

                for category, count in cats:
                    excess = count - self.policy.max_per_category
                    # Delete oldest in this category
                    cursor = db.execute("""
                        DELETE FROM memories
                        WHERE category = ? AND id IN (
                            SELECT id FROM memories
                            WHERE category = ?
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                    """, (category, category, excess))
                    deleted += cursor.rowcount

            db.commit()

        if deleted:
            logger.info("Cleaned up long-term memories", extra={"deleted": deleted})
        return deleted

    def run_all_cleanup(self) -> dict:
        """Run all cleanup operations."""
        return {
            "conversations_deleted": self.cleanup_conversations(),
            "long_term_deleted": self.cleanup_long_term(),
        }


def get_categories() -> list[str]:
    """Standard memory categories for consistent extraction."""
    return [
        "Project",       # Software projects, development work
        "Preference",    # User likes/dislikes, settings
        "Person",        # People mentioned (colleagues, contacts)
        "Fact",          # General knowledge, learned facts
        "Task",          # Todos, commitments, action items
        "Goal",          # Objectives, targets
        "Tool",          # Tools, libraries, frameworks used
        "Configuration", # Settings, configs, environment
        "Issue",         # Problems, bugs, blockers
        "Decision",      # Choices made, rationale
        "Summary",       # Consolidated conversation summaries
    ]


def categorize_fact(fact: str, categories: list[str] = None) -> str:
    """Auto-categorize a fact using simple keyword matching."""
    categories = categories or get_categories()
    fact_lower = fact.lower()

    # Simple keyword-based categorization
    keywords = {
        "Project": ["project", "build", "develop", "code", "repo", "github", "application", "app"],
        "Preference": ["prefer", "like", "dislike", "favorite", "hate", "love", "enjoy"],
        "Person": ["my team", "colleague", "manager", "client", "friend", "boss", "coworker"],
        "Fact": ["is a", "was released", "version", "equals", "means", "stands for"],
        "Task": ["need to", "should", "must", "todo", "task", "action item", "follow up"],
        "Goal": ["goal", "objective", "target", "aim", "want to achieve", "plan to"],
        "Tool": ["tool", "library", "framework", "package", "npm", "pip", "install", "docker", "kubernetes"],
        "Configuration": ["config", "setting", "environment", "env var", "parameter", "option"],
        "Issue": ["bug", "error", "issue", "problem", "fail", "broken", "crash", "exception"],
        "Decision": ["decided", "choose", "chosen", "will use", "going with", "selected"],
    }

    for cat, words in keywords.items():
        if any(w in fact_lower for w in words):
            return cat

    return "Fact"  # Default


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Memory consolidation and cleanup")
    parser.add_argument("--consolidate", action="store_true", help="Run consolidation")
    parser.add_argument("--cleanup", action="store_true", help="Run TTL cleanup")
    parser.add_argument("--threshold", type=int, default=30, help="Consolidation threshold")
    parser.add_argument("--auto", action="store_true", help="Run auto-consolidate if needed")

    args = parser.parse_args()

    if args.consolidate or args.auto:
        consolidator = MemoryConsolidator()
        if args.auto:
            result = consolidator.auto_consolidate_if_needed(args.threshold)
        else:
            result = consolidator.consolidate()
        if result:
            print(f"Consolidated: {result.messages_processed} messages, {result.memories_stored} memories stored")
        else:
            print("No consolidation needed")

    if args.cleanup:
        janitor = MemoryJanitor()
        result = janitor.run_all_cleanup()
        print(f"Cleanup: {result['conversations_deleted']} conversations, {result['long_term_deleted']} long-term memories deleted")