"""Errors that the CLI reports as a usage failure (exit code 2)."""


class PromptdiffError(Exception):
    """The eval files or thresholds are invalid."""
