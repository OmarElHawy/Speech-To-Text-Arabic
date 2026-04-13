"""Logging configuration utilities (T017)"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_json: bool = False
) -> None:
    """
    Setup structured logging with console and optional file handlers
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file name
        log_dir: Directory for log files
        use_json: Use JSON format for logs (requires python-json-logger)
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: [TIMESTAMP] [LEVEL] [MODULE] Message
    console_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file_path = Path(log_dir) / f"{log_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        
        if use_json:
            try:
                from pythonjsonlogger import jsonlogger
                file_format = jsonlogger.JsonFormatter(
                    '%(timestamp)s %(level)s %(name)s %(message)s'
                )
            except ImportError:
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
        else:
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Get logger for a module"""
    return logging.getLogger(name)


# Easy setup for common use cases
def setup_simple_logging(level: str = "INFO") -> None:
    """Simple logging setup with just console output"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    setup_logging(level=level_map.get(level.upper(), logging.INFO))
