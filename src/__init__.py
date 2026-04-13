"""
Arabic Speech-to-Text System - Main package

Core modules for ASR, speaker identification, emotion detection, and keyword spotting.
"""

__version__ = "0.1.0"
__author__ = "Speech-to-Text Team"
__license__ = "MIT"

from .cli import cli_entry_point


def get_app():
    """Get the main application instance"""
    # Placeholder for future app factory
    return None

__all__ = ["get_app", "cli_entry_point"]
