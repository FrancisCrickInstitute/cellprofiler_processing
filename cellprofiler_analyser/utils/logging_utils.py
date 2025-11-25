"""
Logging utilities for Cell Painting processor
"""

import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the package
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
    
    Returns:
        logger: Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger('cellpainting_processor')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name (usually __name__)
    
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(f'cellpainting_processor.{name}')


# Set up default logging
setup_logging()