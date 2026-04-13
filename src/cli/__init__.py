"""CLI package - Command-line interface"""

from .commands import cli

__all__ = ["cli", "transcribe", "benchmark", "evaluate", "demo", "train"]


def cli_entry_point():
    """Entry point for the CLI application"""
    cli()
