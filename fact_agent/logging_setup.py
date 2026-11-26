"""Logging configuration for Fact Agent."""

import logging
import logging.handlers
import sys
from pathlib import Path
from fact_agent.config import get_settings


def setup_logging(settings=None) -> logging.Logger:
    """Configure application logging."""
    s = settings or get_settings()

    # Root logger
    root = logging.getLogger()
    root.setLevel(s.log_level)

    # Clear existing handlers
    root.handlers.clear()

    # Formatter
    formatter = logging.Formatter(s.log_format, datefmt="%H:%M:%S")

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(s.log_level)
    root.addHandler(console)

    # File handler (optional)
    if s.log_file:
        s.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            s.log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(s.log_level)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logger = logging.getLogger("fact_agent")
    logger.info("Logging initialized", extra={"level": s.log_level})
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return logging.getLogger(f"fact_agent.{name}")