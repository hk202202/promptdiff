import json
from pathlib import Path

import pytest

from promptdiff.errors import PromptdiffError
from promptdiff.load import load_jsonl
from tests.helpers import record, write_jsonl


def test_loads_records_and_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        "\n" + json.dumps(record("a", 0.5, latency_ms=12, cost=0)) + "\n\n  \n",
        encoding="utf-8",
    )
    cases = load_jsonl(path)
    assert list(cases) == ["a"]
    assert cases["a"].score == 0.5
    assert cases["a"].latency_ms == 12
    assert cases["a"].cost == 0
    assert cases["a"].prompt == "prompt a"


def test_integer_id_is_stored_as_decimal_text(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({**record("ignored"), "id": 7}) + "\n", encoding="utf-8")
    assert list(load_jsonl(path)) == ["7"]


def test_ignores_unknown_fields(tmp_path: Path):
    path = write_jsonl(tmp_path, "run.jsonl", [record("a", model="demo")])
    assert load_jsonl(path)["a"].id == "a"


def test_missing_file(tmp_path: Path):
    with pytest.raises(PromptdiffError, match="no such file"):
        load_jsonl(tmp_path / "missing.jsonl")


def test_directory_is_rejected(tmp_path: Path):
    with pytest.raises(PromptdiffError, match="is a directory"):
        load_jsonl(tmp_path)


def test_invalid_json(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PromptdiffError, match=r"run.jsonl:1: invalid JSON"):
        load_jsonl(path)


def test_non_object_line(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(PromptdiffError, match="expected a JSON object"):
        load_jsonl(path)


def test_missing_required_field(tmp_path: Path):
    row = record("a")
    del row["cost"]
    path = write_jsonl(tmp_path, "run.jsonl", [row])
    with pytest.raises(PromptdiffError, match="missing required field 'cost'"):
        load_jsonl(path)


def test_duplicate_id(tmp_path: Path):
    path = write_jsonl(tmp_path, "run.jsonl", [record("a"), record("a", 0.2)])
    with pytest.raises(PromptdiffError, match="duplicate id 'a'"):
        load_jsonl(path)


def test_strips_id_and_rejects_empty(tmp_path: Path):
    path = write_jsonl(tmp_path, "run.jsonl", [record("  a  ")])
    assert load_jsonl(path)["a"].prompt.startswith("prompt")
    empty = write_jsonl(tmp_path, "empty.jsonl", [record("   ")])
    with pytest.raises(PromptdiffError, match="id must not be empty"):
        load_jsonl(empty)


def test_rejects_bool_nan_and_negative_metrics(tmp_path: Path):
    boolean = tmp_path / "bool.jsonl"
    boolean.write_text(json.dumps({**record("a"), "score": True}) + "\n", encoding="utf-8")
    with pytest.raises(PromptdiffError, match="score must be a number"):
        load_jsonl(boolean)

    nan = tmp_path / "nan.jsonl"
    nan.write_text(
        '{"id": "a", "prompt": "p", "output": "o", "score": NaN, "latency_ms": 1, "cost": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptdiffError, match="score must be finite"):
        load_jsonl(nan)

    negative = write_jsonl(tmp_path, "neg.jsonl", [record("a", latency_ms=-1)])
    with pytest.raises(PromptdiffError, match="latency_ms must be >= 0"):
        load_jsonl(negative)


def test_rejects_non_utf8(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(PromptdiffError, match="not valid UTF-8"):
        load_jsonl(path)
