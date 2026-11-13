"""Shared fixtures for promptdiff tests."""

from __future__ import annotations

import json
from pathlib import Path

from promptdiff.load import EvalCase

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(directory: Path, name: str, rows: list[dict]) -> Path:
    path = directory / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def record(case_id: str, score: float = 1.0, **overrides: object) -> dict:
    row: dict[str, object] = {
        "id": case_id,
        "prompt": f"prompt {case_id}",
        "output": f"output {case_id}",
        "score": score,
        "latency_ms": 100,
        "cost": 0.01,
    }
    row.update(overrides)
    return row


def case(case_id: str, score: float = 1.0, **overrides: object) -> EvalCase:
    row = record(case_id, score, **overrides)
    return EvalCase(
        id=str(row["id"]),
        prompt=str(row["prompt"]),
        output=str(row["output"]),
        score=float(row["score"]),  # type: ignore[arg-type]
        latency_ms=float(row["latency_ms"]),  # type: ignore[arg-type]
        cost=float(row["cost"]),  # type: ignore[arg-type]
    )
