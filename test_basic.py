"""Basic tests for Fact Agent components."""

import os
import sys
import tempfile
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fact_agent.config import get_settings, Settings
from fact_agent.memory_extractor import store_memories, extract_memories
from fact_agent.memory_retriever import keyword_retrieval, retrieve_memories
from fact_agent.http_client import LLMClient
from fact_agent.exceptions import LLMAPIError
from fact_agent.vector_memory import VectorMemory, _local_embedding


def test_config_loads():
    """Test configuration loads from environment."""
    settings = get_settings()
    assert settings.llm_api_url == "http://localhost:3001/v1/chat/completions"
    assert settings.llm_api_key == "freellmapi-71d7740da25ecc2587b5c49f15e89fbaff5fc9f948b922dc"
    assert settings.conversation_limit == 20
    assert settings.memory_retrieval_limit == 8
    print("✓ Config loads correctly")


def test_settings_dataclass():
    """Test Settings dataclass with custom values."""
    s = Settings(
        llm_api_url="https://api.example.com",
        llm_api_key="test-key",
        conversation_limit=10,
    )
    assert s.llm_api_url == "https://api.example.com"
    assert s.conversation_limit == 10
    print("✓ Settings dataclass works")


def test_memory_db_init():
    """Test memory database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        from fact_agent.agent import initialize_memory
        initialize_memory(db_path)

        # Verify table exists
        with sqlite3.connect(db_path) as db:
            cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "conversations" in tables
            # Check schema
            cursor = db.execute("PRAGMA table_info(conversations)")
            cols = {row[1]: row[2] for row in cursor.fetchall()}
            assert cols["id"] == "INTEGER"
            assert cols["role"] == "TEXT"
            assert cols["content"] == "TEXT"
    print("✓ Memory DB initialization works")


def test_memory_save_load():
    """Test saving and loading conversation memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        from fact_agent.agent import initialize_memory, save_memory, load_memory
        initialize_memory(db_path)

        save_memory("user", "Hello", db_path)
        save_memory("assistant", "Hi there!", db_path)
        save_memory("user", "How are you?", db_path)

        history = load_memory(limit=10, db_path=db_path)
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "How are you?"

        # Test limit
        history = load_memory(limit=2, db_path=db_path)
        assert len(history) == 2
        assert history[0]["content"] == "Hi there!"  # Most recent 2
    print("✓ Memory save/load works")


def test_long_term_memory_store():
    """Test storing memories in long-term DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ltm.db"
        # Create schema
        with sqlite3.connect(db_path) as db:
            db.execute("""
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        memories = [
            {"category": "Project", "fact": "User is building Fact Agent"},
            {"category": "Preference", "fact": "User prefers Python"},
        ]
        stored = store_memories(memories, str(db_path))
        assert stored == 2

        # Test deduplication
        stored = store_memories(memories, str(db_path))
        assert stored == 0  # Duplicates not stored

        # Verify data
        with sqlite3.connect(db_path) as db:
            rows = db.execute("SELECT category, fact FROM memories").fetchall()
            assert len(rows) == 2
    print("✓ Long-term memory store works")


def test_keyword_retrieval():
    """Test keyword-based memory retrieval."""
    # Rows should be ordered by ID DESC (most recent first) as retrieve_memories does
    rows = [
        (3, "Fact", "Python is a programming language"),
        (2, "Preference", "User prefers dark mode"),
        (1, "Project", "User is building Fact Agent in Python"),
    ]

    # Query matching multiple
    result = keyword_retrieval(rows, "Python", limit=5)
    assert "Fact Agent" in result
    assert "programming language" in result
    assert "dark mode" not in result

    # Empty query returns most recent
    result = keyword_retrieval(rows, "", limit=2)
    assert "Python is a programming language" in result
    assert "User prefers dark mode" in result

    # No matches
    result = keyword_retrieval(rows, "nonexistent", limit=5)
    assert result == ""
    print("✓ Keyword retrieval works")


def test_llm_client_creation():
    """Test LLM client instantiation."""
    client = LLMClient(
        api_url="http://localhost:3001/v1/chat/completions",
        api_key="test-key",
        model="test-model",
    )
    assert client.api_url == "http://localhost:3001/v1/chat/completions"
    assert client.api_key == "test-key"
    assert client.model == "test-model"
    print("✓ LLM client creation works")


def test_vector_memory():
    """Test vector memory with local embeddings."""
    from fact_agent.vector_memory import VectorMemory, _local_embedding

    # Test local embedding generation
    emb1 = _local_embedding("Python programming language")
    emb2 = _local_embedding("Python coding")
    emb3 = _local_embedding("unrelated topic xyz")

    assert len(emb1) == 256
    assert len(emb2) == 256

    # Similar texts should have higher similarity
    from fact_agent.vector_memory import VectorMemory
    vm = VectorMemory(use_local_fallback=True)
    sim1 = vm._cosine_similarity(emb1, emb2)
    sim2 = vm._cosine_similarity(emb1, emb3)
    assert sim1 > sim2, f"Similar texts should score higher: {sim1} vs {sim2}"
    print("✓ Local embeddings work")

    # Test VectorMemory search
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_vector.db"
        vm = VectorMemory(db_path=db_path, use_local_fallback=True)

        # Add test memories
        vm.add_memory(1, "Project", "User is building Fact Agent in Python")
        vm.add_memory(2, "Fact", "Python is a programming language")
        vm.add_memory(3, "Preference", "User prefers dark mode")

        # Search
        results = vm.search("Python", limit=2)
        assert len(results) == 2
        assert "Python" in results[0]["fact"] or "programming language" in results[0]["fact"]

        results = vm.search("dark mode", limit=2)
        assert len(results) >= 1
        assert "dark mode" in results[0]["fact"].lower()
    print("✓ Vector memory search works")


def test_tools():
    """Test tool registry and tools."""
    from fact_agent.tools import WebSearchTool, CalculatorTool, FileOpsTool, ToolRegistry, ToolResult

    # Test calculator
    calc = CalculatorTool()
    result = calc.execute(expression='2 * 3.14159 * 5')
    assert result.success
    assert "31.4159" in result.content
    print("✓ Calculator tool works")

    # Test file ops
    with tempfile.TemporaryDirectory() as tmpdir:
        file_ops = FileOpsTool(Path(tmpdir))
        result = file_ops.execute(operation='write', path='test.txt', content='Hello World')
        assert result.success
        result = file_ops.execute(operation='read', path='test.txt')
        assert result.success
        assert result.content == 'Hello World'
        result = file_ops.execute(operation='list', path='.')
        assert result.success
        assert 'test.txt' in result.content
    print("✓ File ops tool works")

    # Test tool registry
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert len(tools) == 3  # web_search, calculator, file_ops
    names = [t.name for t in tools]
    assert 'web_search' in names
    assert 'calculator' in names
    assert 'file_ops' in names
    specs = registry.get_function_specs()
    assert len(specs) == 3
    print("✓ Tool registry works")

    # Test web search (may be slow, just check it doesn't error)
    # Skipping actual web search in tests to keep them fast
    print("✓ Tools module imports work")


def run_all_tests():
    """Run all tests."""
    print("Running basic tests...\n")
    test_config_loads()
    test_settings_dataclass()
    test_memory_db_init()
    test_memory_save_load()
    test_long_term_memory_store()
    test_keyword_retrieval()
    test_llm_client_creation()
    test_vector_memory()
    test_tools()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()