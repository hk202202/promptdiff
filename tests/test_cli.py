import json
from pathlib import Path

import pytest

from promptdiff.cli import main
from tests.helpers import ROOT, record, write_jsonl


def test_example_command_exits_nonzero_and_prints_the_report(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    code = main(["examples/base.jsonl", "examples/new.jsonl"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == ""
    assert "translate" in captured.out
    assert "newly failing" in captured.out
    assert "dropped-case" in captured.out
    assert "added-case" in captured.out
    assert "FAIL" in captured.out
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert captured.out.strip() in readme


def test_raised_limits_exit_zero(tmp_path: Path, capsys):
    base = write_jsonl(tmp_path, "base.jsonl", [record("a", 1.0), record("b", 1.0)])
    new = write_jsonl(tmp_path, "new.jsonl", [record("a", 0.0), record("b", 0.4)])
    code = main(
        [
            str(base),
            str(new),
            "--max-regressions",
            "2",
            "--max-new-failures",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "OK" in captured.out
    assert "within thresholds" in captured.out


def test_json_format_is_machine_readable(tmp_path: Path, capsys):
    base = write_jsonl(tmp_path, "base.jsonl", [record("a", 1, latency_ms=10, cost=0.1)])
    new = write_jsonl(tmp_path, "new.jsonl", [record("a", 1, latency_ms=40, cost=0.1)])
    code = main(
        [str(base), str(new), "--format", "json", "--max-latency-increase", "10"]
    )
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["summary"]["failed"] is True
    assert "mean latency increased" in payload["summary"]["fail_reasons"][0]
    assert payload["cases"][0]["latency_delta_ms"] == 30


def test_invalid_input_exits_2(tmp_path: Path, capsys):
    missing = main([str(tmp_path / "nope.jsonl"), str(tmp_path / "also.jsonl")])
    captured = capsys.readouterr()
    assert missing == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")

    base = write_jsonl(tmp_path, "base.jsonl", [record("a")])
    code = main([str(base), str(base), "--min-score-drop", "-1"])
    captured = capsys.readouterr()
    assert code == 2
    assert "min score drop" in captured.err


def test_help_and_version():
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0


def test_show_all_flag(tmp_path: Path, capsys):
    base = write_jsonl(tmp_path, "base.jsonl", [record("a", latency_ms=50, cost=0.05)])
    new = write_jsonl(tmp_path, "new.jsonl", [record("a", latency_ms=20, cost=0.01)])
    code = main([str(base), str(new), "--show-all"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Latency decreases" in captured.out
    assert "Cost decreases" in captured.out
