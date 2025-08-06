"""
Logging utility for CourseMate RAG Application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config.settings import settings


def setup_logger(
    name: str = "coursemate",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and configure logger."""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or getattr(logging, settings.environment.value.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger() 