"""Load JSONL eval runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from promptdiff.errors import PromptdiffError

_REQUIRED = ("id", "prompt", "output", "score", "latency_ms", "cost")


@dataclass(frozen=True)
class EvalCase:
    """One scored prompt from an eval run."""

    id: str
    prompt: str
    output: str
    score: float
    latency_ms: float
    cost: float


def load_jsonl(path: str | Path) -> dict[str, EvalCase]:
    """Read a JSONL eval file into an insertion-ordered id map.

    Blank lines are ignored. Ids are compared as strings; integer ids are
    accepted and rendered in decimal. Duplicate ids are an error.
    """

    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise PromptdiffError(f"{file_path}: no such file") from exc
    except IsADirectoryError as exc:
        raise PromptdiffError(f"{file_path}: is a directory") from exc
    except OSError as exc:
        message = exc.strerror or str(exc)
        raise PromptdiffError(f"{file_path}: {message}") from exc
    except UnicodeDecodeError as exc:
        raise PromptdiffError(f"{file_path}: not valid UTF-8") from exc

    cases: dict[str, EvalCase] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromptdiffError(
                f"{file_path}:{line_no}: invalid JSON ({exc.msg})"
            ) from exc
        case = _parse_case(file_path, line_no, payload)
        if case.id in cases:
            raise PromptdiffError(
                f"{file_path}:{line_no}: duplicate id {case.id!r}"
            )
        cases[case.id] = case
    return cases


def _parse_case(path: Path, line_no: int, payload: object) -> EvalCase:
    if not isinstance(payload, dict):
        raise PromptdiffError(f"{path}:{line_no}: expected a JSON object")
    missing = [field for field in _REQUIRED if field not in payload]
    if missing:
        raise PromptdiffError(
            f"{path}:{line_no}: missing required field {missing[0]!r}"
        )
    return EvalCase(
        id=_parse_id(path, line_no, payload["id"]),
        prompt=_parse_text(path, line_no, "prompt", payload["prompt"]),
        output=_parse_text(path, line_no, "output", payload["output"]),
        score=_parse_number(path, line_no, "score", payload["score"], nonnegative=False),
        latency_ms=_parse_number(
            path, line_no, "latency_ms", payload["latency_ms"], nonnegative=True
        ),
        cost=_parse_number(path, line_no, "cost", payload["cost"], nonnegative=True),
    )


def _parse_id(path: Path, line_no: int, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PromptdiffError(f"{path}:{line_no}: id must be a string or integer")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PromptdiffError(f"{path}:{line_no}: id must not be empty")
        return text
    return str(value)


def _parse_text(path: Path, line_no: int, field: str, value: object) -> str:
    if not isinstance(value, str):
        raise PromptdiffError(f"{path}:{line_no}: {field} must be a string")
    return value


def _parse_number(
    path: Path,
    line_no: int,
    field: str,
    value: object,
    *,
    nonnegative: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromptdiffError(f"{path}:{line_no}: {field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PromptdiffError(f"{path}:{line_no}: {field} must be finite")
    if nonnegative and number < 0:
        raise PromptdiffError(f"{path}:{line_no}: {field} must be >= 0")
    return number
