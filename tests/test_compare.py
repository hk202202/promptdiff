import json

import pytest

from promptdiff.compare import Thresholds, compare_cases, compare_paths
from promptdiff.errors import PromptdiffError
from tests.helpers import ROOT, case, record


def test_classifies_score_changes_and_new_failures():
    report = compare_cases(
        {
            "drop": case("drop", 0.9),
            "fail": case("fail", 1.0),
            "gain": case("gain", 0.4),
            "same": case("same", 1.0),
            "already": case("already", 0.2),
            "gone": case("gone", 1.0),
        },
        {
            "drop": case("drop", 0.7),
            "fail": case("fail", 0.0),
            "gain": case("gain", 0.9),
            "same": case("same", 1.0),
            "already": case("already", 0.0),
            "fresh": case("fresh", 1.0),
        },
    )
    by_id = {item.id: item for item in report.cases}
    assert [item.id for item in report.cases] == [
        "drop",
        "fail",
        "gain",
        "same",
        "already",
        "gone",
        "fresh",
    ]
    assert by_id["fail"].is_newly_failing(report.thresholds)
    assert by_id["fail"].is_score_regression(report.thresholds)
    assert by_id["already"].still_failing(report.thresholds)
    assert not by_id["already"].is_newly_failing(report.thresholds)
    assert by_id["gain"].is_improved(report.thresholds)
    assert by_id["gone"].presence == "removed"
    assert by_id["fresh"].presence == "added"
    assert report.summary.shared == 5
    assert report.summary.regressions == 3
    assert report.summary.new_failures == 1
    assert report.summary.improved == 1
    assert report.summary.unchanged == 1
    assert report.summary.added == 1
    assert report.summary.removed == 1
    assert report.summary.regressions + report.summary.improved + report.summary.unchanged == 5
    assert report.summary.failed is True


def test_drop_equal_to_the_minimum_is_not_a_regression():
    report = compare_cases(
        {"a": case("a", 0.9)},
        {"a": case("a", 0.7)},
        Thresholds(min_score_drop=0.2),
    )
    assert report.summary.regressions == 0
    assert report.summary.unchanged == 1
    assert report.summary.failed is False


def test_new_failure_inside_the_drop_tolerance_still_fails_the_gate():
    report = compare_cases(
        {"a": case("a", 0.5)},
        {"a": case("a", 0.49)},
        Thresholds(min_score_drop=0.1, max_regressions=5, max_new_failures=0),
    )
    assert report.summary.regressions == 0
    assert report.summary.new_failures == 1
    assert report.summary.fail_reasons == ("1 newly failing case exceeds max 0",)


def test_counts_at_the_limit_pass():
    report = compare_cases(
        {"a": case("a", 1.0), "b": case("b", 1.0)},
        {"a": case("a", 0.0), "b": case("b", 0.0)},
        Thresholds(max_regressions=2, max_new_failures=2),
    )
    assert report.summary.regressions == 2
    assert report.summary.failed is False
    assert report.summary.fail_reasons == ()


def test_removed_cases_are_reported_and_gated_only_when_asked():
    base = {"keep": case("keep"), "gone": case("gone")}
    new = {"keep": case("keep")}
    open_gate = compare_cases(base, new)
    assert open_gate.summary.removed == 1
    assert open_gate.summary.failed is False
    closed = compare_cases(base, new, Thresholds(max_removed=0))
    assert closed.summary.fail_reasons == ("1 removed case exceeds max 0",)


def test_latency_and_cost_use_shared_pairs_and_percent_gates():
    report = compare_cases(
        {
            "a": case("a", latency_ms=100, cost=0.02),
            "b": case("b", latency_ms=300, cost=0.02),
            "only-base": case("only-base", latency_ms=9999, cost=9),
        },
        {
            "a": case("a", latency_ms=150, cost=0.03),
            "b": case("b", latency_ms=300, cost=0.03),
            "only-new": case("only-new", latency_ms=1, cost=0),
        },
        Thresholds(max_latency_increase_pct=20, max_cost_increase_pct=25),
    )
    assert report.summary.mean_latency_base_ms == pytest.approx(200)
    assert report.summary.mean_latency_new_ms == pytest.approx(225)
    assert report.summary.mean_latency_increase_pct == pytest.approx(12.5)
    assert report.summary.total_cost_base == pytest.approx(0.04)
    assert report.summary.total_cost_new == pytest.approx(0.06)
    assert report.summary.total_cost_increase_pct == pytest.approx(50)
    assert report.summary.fail_reasons == ("total cost increased 50.00% (max 25%)",)


def test_increase_from_zero_fails_a_finite_percent_gate():
    report = compare_cases(
        {"a": case("a", latency_ms=0, cost=0)},
        {"a": case("a", latency_ms=10, cost=0.01)},
        Thresholds(max_latency_increase_pct=1000, max_cost_increase_pct=1000),
    )
    assert report.summary.mean_latency_from_zero is True
    assert report.summary.total_cost_from_zero is True
    assert report.summary.fail_reasons == (
        "mean latency increased from 0 (max 1000%)",
        "total cost increased from 0 (max 1000%)",
    )


def test_mean_score_ignores_added_and_removed_cases():
    report = compare_cases(
        {"keep": case("keep", 1.0), "gone": case("gone", 0.0)},
        {"keep": case("keep", 0.5), "fresh": case("fresh", 1.0)},
    )
    assert report.summary.mean_score_base == pytest.approx(1.0)
    assert report.summary.mean_score_new == pytest.approx(0.5)
    assert report.summary.mean_score_delta == pytest.approx(-0.5)


def test_invalid_thresholds_are_rejected():
    with pytest.raises(PromptdiffError, match="min score drop"):
        Thresholds(min_score_drop=-0.1)
    with pytest.raises(PromptdiffError, match="pass score"):
        Thresholds(pass_score=float("nan"))


def test_numeric_and_string_ids_join(tmp_path):
    base = tmp_path / "base.jsonl"
    new = tmp_path / "new.jsonl"
    base.write_text(
        json.dumps({**record("placeholder"), "id": 7, "score": 1}) + "\n",
        encoding="utf-8",
    )
    new.write_text(
        json.dumps({**record("placeholder"), "id": "7", "score": 0}) + "\n",
        encoding="utf-8",
    )
    report = compare_paths(base, new)
    assert [item.id for item in report.cases] == ["7"]
    assert report.summary.shared == 1
    assert report.summary.regressions == 1


def test_example_files_have_the_documented_regressions():
    report = compare_paths(ROOT / "examples" / "base.jsonl", ROOT / "examples" / "new.jsonl")
    regressions = sorted(
        case.id for case in report.cases if case.is_score_regression(report.thresholds)
    )
    failures = [case.id for case in report.cases if case.is_newly_failing(report.thresholds)]
    assert regressions == ["cite-year", "summarize", "translate"]
    assert failures == ["translate"]
    assert report.summary.shared == 6
    assert report.summary.added == 1
    assert report.summary.removed == 1
    assert report.summary.improved == 1
    assert report.summary.mean_score_delta == pytest.approx(-0.2)
    assert report.summary.failed is True
