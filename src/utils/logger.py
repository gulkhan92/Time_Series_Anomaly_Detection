"""
Structured logging utility.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import os

def setup_logger(
    name: str = "anomaly_detection",
    level: str = os.getenv("LOG_LEVEL", "INFO"),
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Avoid dup handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
