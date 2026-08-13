"""
Logging and configuration utilities for ResearchRAG.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DIR = PROCESSED_DIR / "cache"
INDEX_DIR = PROCESSED_DIR / "indices"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
for directory in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, SAMPLE_DIR, CACHE_DIR, INDEX_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

# Logger setup
def get_logger(name: str = "ResearchRAG") -> logging.Logger:
    """Returns a standardized logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = get_logger()
