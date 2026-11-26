"""Configuration management for Fact Agent (stdlib + requests only)."""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings loaded from environment variables and .env file."""

    # LLM API
    llm_api_url: str = "http://localhost:3001/v1/chat/completions"
    llm_api_key: str = ""
    llm_model: str = "auto"
    llm_timeout: int = 60

    # Memory
    memory_db_path: Path = field(default_factory=lambda: Path("memory.db"))
    long_term_memory_db_path: Path = field(default_factory=lambda: Path("long_term_memory.db"))
    conversation_limit: int = 20
    memory_retrieval_limit: int = 8
    max_fact_length: int = 500

    # JANCTION Cloud Rendering
    janction_api_base: str = "https://render.janction.jp"
    janction_api_key_path: Path = field(default_factory=lambda: Path.home() / ".config" / "fact-agent" / "janction_api_key.txt")
    janction_timeout: int = 30

    # Gemini (optional)
    gemini_api_key: str = ""
    gemini_api_key_path: Path = field(default_factory=lambda: Path("gemini_api_key.txt"))

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_file: Path | None = None

    # Agent behavior
    system_prompt: str = """You are a general-purpose fact-based AI agent.

Your job is to:
1. Understand the user's request.
2. Separate factual claims from opinions or creative content.
3. Never knowingly invent facts, sources, quotes, statistics, or events.
4. Clearly identify uncertainty when information cannot be verified.
5. Give useful, clear answers.
6. Distinguish verified information from inference."""
    enable_semantic_retrieval: bool = True
    enable_memory_extraction: bool = True
    enable_vector_retrieval: bool = True

    def __post_init__(self):
        # Expand paths
        for attr in ["memory_db_path", "long_term_memory_db_path", "janction_api_key_path", "gemini_api_key_path", "log_file"]:
            val = getattr(self, attr)
            if val and isinstance(val, str):
                setattr(self, attr, Path(val).expanduser().resolve())
            elif val and isinstance(val, Path):
                setattr(self, attr, val.expanduser().resolve())

        # Validate log level
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        self.log_level = self.log_level.upper()


def _load_dotenv() -> dict[str, str]:
    """Load .env file into environment (simple implementation)."""
    env_path = Path(".env")
    if not env_path.is_file():
        return {}

    result = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with .env fallback."""
    # Check actual env first
    val = os.environ.get(key)
    if val is not None:
        return val
    # Check .env cache
    return _DOTENV_CACHE.get(key, default)


# Load .env once
_DOTENV_CACHE = _load_dotenv()


def _parse_bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes", "on")


def _parse_int(val: str, default: int) -> int:
    try:
        return int(val)
    except ValueError:
        return default


def _parse_optional_int(val: str) -> int | None:
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance loaded from environment."""
    return Settings(
        llm_api_url=_get_env("LLM_API_URL"),
        llm_api_key=_get_env("LLM_API_KEY"),
        llm_model=_get_env("LLM_MODEL"),
        llm_timeout=_parse_int(_get_env("LLM_TIMEOUT"), 60),

        memory_db_path=Path(_get_env("MEMORY_DB_PATH", "memory.db")),
        long_term_memory_db_path=Path(_get_env("LONG_TERM_MEMORY_DB_PATH", "long_term_memory.db")),
        conversation_limit=_parse_int(_get_env("CONVERSATION_LIMIT"), 20),
        memory_retrieval_limit=_parse_int(_get_env("MEMORY_RETRIEVAL_LIMIT"), 8),
        max_fact_length=_parse_int(_get_env("MAX_FACT_LENGTH"), 500),

        janction_api_base=_get_env("JANCTION_API_BASE"),
        janction_api_key_path=Path(_get_env("JANCTION_API_KEY_PATH", str(Path.home() / ".config" / "fact-agent" / "janction_api_key.txt"))),
        janction_timeout=_parse_int(_get_env("JANCTION_TIMEOUT"), 30),

        gemini_api_key=_get_env("GEMINI_API_KEY"),
        gemini_api_key_path=Path(_get_env("GEMINI_API_KEY_PATH", "gemini_api_key.txt")),

        log_level=_get_env("LOG_LEVEL", "INFO"),
        log_format=_get_env("LOG_FORMAT", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        log_file=Path(_get_env("LOG_FILE")) if _get_env("LOG_FILE") else None,

        system_prompt=_get_env("SYSTEM_PROMPT", ""),  # Will use default if empty
        enable_semantic_retrieval=_parse_bool(_get_env("ENABLE_SEMANTIC_RETRIEVAL", "true")),
        enable_memory_extraction=_parse_bool(_get_env("ENABLE_MEMORY_EXTRACTION", "true")),
        enable_vector_retrieval=_parse_bool(_get_env("ENABLE_VECTOR_RETRIEVAL", "true")),
    )


def load_gemini_key_from_file(settings: Settings | None = None) -> str:
    """Load Gemini API key from file if not set in env."""
    s = settings or get_settings()
    if s.gemini_api_key:
        return s.gemini_api_key

    path = s.gemini_api_key_path
    if path.is_file():
        try:
            return path.read_text().strip()
        except Exception:
            pass
    return ""


def load_janction_key(settings: Settings | None = None) -> str:
    """Load JANCTION API key from configured path."""
    s = settings or get_settings()
    path = s.janction_api_key_path
    if not path.is_file():
        raise RuntimeError(f"JANCTION API key file not found: {path}")

    # Check permissions (warn only, don't fail on all platforms)
    try:
        import stat
        permissions = stat.S_IMODE(path.stat().st_mode)
        if permissions & 0o077:
            import logging
            logging.getLogger(__name__).warning(
                "JANCTION API key file permissions are too open (should be 600)"
            )
    except Exception:
        pass

    key = path.read_text().strip()
    if not key:
        raise RuntimeError("JANCTION API key file is empty")
    return key