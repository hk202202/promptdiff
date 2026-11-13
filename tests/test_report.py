import json

from promptdiff.compare import Thresholds, compare_cases
from promptdiff.report import fmt_cost, fmt_score, render_json, render_text
from tests.helpers import case


def test_score_and_cost_formatting_keeps_small_values():
    assert fmt_score(1.0) == "1"
    assert fmt_score(0.0) == "0"
    assert fmt_score(-0.2) == "-0.2"
    assert fmt_score(1e-8) == "1e-08"
    assert fmt_cost(0.0012) == "$0.0012"
    assert fmt_cost(0.004) == "$0.004"
    assert fmt_cost(1.5) == "$1.50"


def test_text_report_marks_failures_and_can_hide_decreases():
    report = compare_cases(
        {"slow": case("slow", 1, latency_ms=100, cost=0.01), "fast": case("fast", 1, latency_ms=80, cost=0.02)},
        {"slow": case("slow", 0, latency_ms=250, cost=0.04), "fast": case("fast", 1, latency_ms=40, cost=0.01)},
    )
    text = render_text(report)
    assert "Score deltas" in text
    assert "regression, newly failing" in text
    assert "Latency increases" in text
    assert "slow" in text
    assert "Latency decreases" not in text
    assert text.endswith("FAIL\n  1 score regression exceeds max 0\n  1 newly failing case exceeds max 0\n")
    shown = render_text(report, show_all=True, verbose=True)
    assert "Latency decreases" in shown
    assert "Cost decreases" in shown
    assert "prompt: prompt slow" in shown
    assert "output slow -> output slow" in shown


def test_passing_report_ends_ok_and_json_matches_summary():
    report = compare_cases(
        {"a": case("a", 0.4, latency_ms=0, cost=0)},
        {"a": case("a", 0.4, latency_ms=0, cost=0)},
        Thresholds(max_regressions=0),
    )
    text = render_text(report)
    assert "OK" in text
    assert "within thresholds" in text
    assert "from 0" not in text
    payload = json.loads(render_json(report))
    assert payload["tool"] == "promptdiff"
    assert payload["summary"]["failed"] is False
    assert payload["summary"]["fail_reasons"] == []
    assert payload["cases"][0]["score_delta"] == 0
    assert payload["cases"][0]["still_failing"] is True


def test_tied_displayed_drops_sort_by_id():
    report = compare_cases(
        {"summarize": case("summarize", 0.9), "cite-year": case("cite-year", 0.2)},
        {"summarize": case("summarize", 0.7), "cite-year": case("cite-year", 0.0)},
    )
    score_section = render_text(report).split("Latency increases")[0]
    assert score_section.index("cite-year") < score_section.index("summarize")


def test_empty_runs_say_there_is_nothing_to_compare():
    report = compare_cases({}, {})
    text = render_text(report)
    assert "no shared cases" in text
    assert "mean score" in text
    assert "n/a" in text
    assert "OK" in text


def test_verbose_shows_both_prompts_when_they_differ():
    report = compare_cases(
        {"a": case("a", 1, prompt="before")},
        {"a": case("a", 0, prompt="after")},
    )
    text = render_text(report, verbose=True)
    assert "prompt base: before" in text
    assert "prompt new:  after" in text
