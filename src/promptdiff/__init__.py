"""Compare LLM eval runs and report regressions."""

from promptdiff.compare import (
    CaseDelta,
    Report,
    Summary,
    Thresholds,
    compare_cases,
    compare_paths,
)
from promptdiff.errors import PromptdiffError
from promptdiff.load import EvalCase, load_jsonl

__version__ = "0.1.0"

__all__ = [
    "CaseDelta",
    "EvalCase",
    "PromptdiffError",
    "Report",
    "Summary",
    "Thresholds",
    "compare_cases",
    "compare_paths",
    "load_jsonl",
    "__version__",
]
