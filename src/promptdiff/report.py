"""Text and JSON renderers for a comparison report."""

from __future__ import annotations

import json
from collections.abc import Sequence

from promptdiff import __version__
from promptdiff.compare import CaseDelta, Report, Thresholds, exceeds

_DETAIL_LIMIT = 160


def render_json(report: Report) -> str:
    """Serialize a report as pretty-printed JSON with a trailing newline."""

    return json.dumps(to_dict(report), indent=2, ensure_ascii=False) + "\n"


def to_dict(report: Report) -> dict:
    """Return a JSON-ready mapping. Cases stay in baseline order."""

    thresholds = report.thresholds
    summary = report.summary
    return {
        "tool": "promptdiff",
        "version": __version__,
        "base": report.base_path,
        "new": report.new_path,
        "thresholds": {
            "pass_score": thresholds.pass_score,
            "min_score_drop": thresholds.min_score_drop,
            "max_regressions": thresholds.max_regressions,
            "max_new_failures": thresholds.max_new_failures,
            "max_removed": thresholds.max_removed,
            "max_latency_increase_pct": thresholds.max_latency_increase_pct,
            "max_cost_increase_pct": thresholds.max_cost_increase_pct,
        },
        "summary": {
            "shared": summary.shared,
            "added": summary.added,
            "removed": summary.removed,
            "mean_score_base": summary.mean_score_base,
            "mean_score_new": summary.mean_score_new,
            "mean_score_delta": summary.mean_score_delta,
            "regressions": summary.regressions,
            "new_failures": summary.new_failures,
            "improved": summary.improved,
            "unchanged": summary.unchanged,
            "mean_latency_base_ms": summary.mean_latency_base_ms,
            "mean_latency_new_ms": summary.mean_latency_new_ms,
            "mean_latency_increase_pct": summary.mean_latency_increase_pct,
            "mean_latency_from_zero": summary.mean_latency_from_zero,
            "latency_pairs": summary.latency_pairs,
            "total_cost_base": summary.total_cost_base,
            "total_cost_new": summary.total_cost_new,
            "total_cost_increase_pct": summary.total_cost_increase_pct,
            "total_cost_from_zero": summary.total_cost_from_zero,
            "cost_pairs": summary.cost_pairs,
            "failed": summary.failed,
            "fail_reasons": list(summary.fail_reasons),
        },
        "cases": [_case_dict(case, thresholds) for case in report.cases],
    }


def render_text(report: Report, *, show_all: bool = False, verbose: bool = False) -> str:
    """Render a human-readable report ending in OK or FAIL."""

    lines: list[str] = [
        f"base  {report.base_path}",
        f"new   {report.new_path}",
        "",
        *_summary_lines(report),
        "",
        *_score_lines(report),
    ]
    new_failures = _sorted_by_drop(
        [case for case in report.cases if case.is_newly_failing(report.thresholds)]
    )
    if new_failures:
        lines.append("")
        lines.extend(_new_failure_lines(new_failures))
    added = [case for case in report.cases if case.presence == "added"]
    removed = [case for case in report.cases if case.presence == "removed"]
    if added or removed:
        lines.append("")
        lines.extend(_membership_lines(added, removed))
    lines.append("")
    lines.extend(_metric_lines(report, kind="latency", show_all=show_all))
    lines.append("")
    lines.extend(_metric_lines(report, kind="cost", show_all=show_all))
    if verbose:
        details = _sorted_by_drop(
            [
                case
                for case in report.cases
                if case.is_score_regression(report.thresholds)
                or case.is_newly_failing(report.thresholds)
            ]
        )
        if details:
            lines.append("")
            lines.extend(_detail_lines(details))
    lines.append("")
    lines.extend(_result_lines(report))
    return "\n".join(lines) + "\n"


def _summary_lines(report: Report) -> list[str]:
    summary = report.summary
    thresholds = report.thresholds
    score = _score_summary(summary.mean_score_base, summary.mean_score_new, summary.mean_score_delta)
    latency = _aggregate_summary(
        summary.mean_latency_base_ms,
        summary.mean_latency_new_ms,
        summary.mean_latency_increase_pct,
        summary.mean_latency_from_zero,
        summary.latency_pairs,
        summary.shared,
        fmt_ms,
        thresholds.max_latency_increase_pct,
    )
    cost = _aggregate_summary(
        summary.total_cost_base,
        summary.total_cost_new,
        summary.total_cost_increase_pct,
        summary.total_cost_from_zero,
        summary.cost_pairs,
        summary.shared,
        fmt_cost,
        thresholds.max_cost_increase_pct,
    )
    removed_limit = ""
    if thresholds.max_removed is not None:
        removed_limit = f" (max removed {thresholds.max_removed})"
    rows = [
        ("shared", str(summary.shared)),
        ("added / removed", f"{summary.added} / {summary.removed}{removed_limit}"),
        ("pass score", fmt_score(thresholds.pass_score)),
        ("mean score", score),
        (
            "score regressions",
            f"{summary.regressions} (max {thresholds.max_regressions}, drop > {fmt_score(thresholds.min_score_drop)})",
        ),
        ("newly failing", f"{summary.new_failures} (max {thresholds.max_new_failures})"),
        ("improved", str(summary.improved)),
        ("unchanged", str(summary.unchanged)),
        ("mean latency", latency),
        ("total cost", cost),
    ]
    width = max(len(label) for label, _ in rows)
    return ["Summary", *(f"  {label.ljust(width)}  {value}" for label, value in rows)]


def _score_summary(base: float | None, new: float | None, delta: float | None) -> str:
    if base is None or new is None or delta is None:
        return "n/a"
    return f"{fmt_score(base)} -> {fmt_score(new)} (Δ {fmt_delta(delta)})"


def _aggregate_summary(
    base: float | None,
    new: float | None,
    pct: float | None,
    from_zero: bool,
    pairs: int,
    shared: int,
    format_value,
    limit: float | None,
) -> str:
    if base is None or new is None:
        return "n/a"
    details: list[str] = []
    if from_zero:
        details.append("from 0")
    elif pct is not None:
        details.append(fmt_pct(pct))
    details.append(f"{pairs} cases" if pairs == shared else f"{pairs} of {shared} cases")
    if limit is not None:
        details.append(f"max +{fmt_score(limit)}%")
    return f"{format_value(base)} -> {format_value(new)} ({', '.join(details)})"


def _score_lines(report: Report) -> list[str]:
    shared = [case for case in report.cases if case.presence == "shared"]
    if not shared:
        return ["Score deltas", "  no shared cases"]
    rows = []
    for case in _sorted_by_drop(shared):
        assert case.base_score is not None
        assert case.new_score is not None
        assert case.score_delta is not None
        rows.append(
            [
                case.id,
                fmt_score(case.base_score),
                fmt_score(case.new_score),
                fmt_delta(case.score_delta),
                _note(case, report.thresholds),
            ]
        )
    return ["Score deltas", _table(["id", "base", "new", "delta", "note"], rows)]


def _new_failure_lines(cases: Sequence[CaseDelta]) -> list[str]:
    rows = [
        [case.id, f"{fmt_score(case.base_score or 0)} -> {fmt_score(case.new_score or 0)}"]
        for case in cases
    ]
    return ["Newly failing", _table(["id", "score"], rows)]


def _membership_lines(added: Sequence[CaseDelta], removed: Sequence[CaseDelta]) -> list[str]:
    lines = ["Case coverage"]
    if removed:
        lines.append("  removed")
        lines.extend(f"    {case.id}" for case in removed)
    if added:
        lines.append("  added")
        lines.extend(f"    {case.id}" for case in added)
    return lines


def _metric_lines(report: Report, *, kind: str, show_all: bool) -> list[str]:
    shared = [case for case in report.cases if case.presence == "shared"]
    if kind == "latency":
        title = "Latency"
        format_value = fmt_ms
        format_delta = fmt_signed_ms

        def measurement(case: CaseDelta) -> tuple[float | None, float | None, float | None]:
            return case.base_latency_ms, case.new_latency_ms, case.latency_delta_ms
    else:
        title = "Cost"
        format_value = fmt_cost
        format_delta = fmt_signed_cost

        def measurement(case: CaseDelta) -> tuple[float | None, float | None, float | None]:
            return case.base_cost, case.new_cost, case.cost_delta

    increases = []
    decreases = []
    for case in shared:
        base, new, delta = measurement(case)
        if delta is None or base is None or new is None or not exceeds(abs(delta), 0):
            continue
        row = [
            case.id,
            format_value(base),
            format_value(new),
            _delta_with_pct(delta, base, format_delta),
        ]
        if exceeds(delta, 0):
            increases.append((delta, case.id, row))
        elif exceeds(0, delta):
            decreases.append((delta, case.id, row))
    increases.sort(key=lambda item: (-item[0], item[1]))
    decreases.sort(key=lambda item: (item[0], item[1]))
    lines = [f"{title} increases"]
    if increases:
        lines.append(_table(["id", "base", "new", "delta"], [item[2] for item in increases]))
    else:
        lines.append("  no increases")
    if show_all:
        lines.append(f"{title} decreases")
        if decreases:
            lines.append(_table(["id", "base", "new", "delta"], [item[2] for item in decreases]))
        else:
            lines.append("  no decreases")
    return lines


def _detail_lines(cases: Sequence[CaseDelta]) -> list[str]:
    lines = ["Details"]
    for case in cases:
        lines.append(f"  {case.id}")
        prompts_differ = (
            case.base_prompt is not None
            and case.new_prompt is not None
            and case.base_prompt != case.new_prompt
        )
        if prompts_differ:
            lines.append(f"    prompt base: {_shorten(case.base_prompt)}")
            lines.append(f"    prompt new:  {_shorten(case.new_prompt)}")
        else:
            lines.append(f"    prompt: {_shorten(case.prompt)}")
        lines.append(
            f"    output: {_shorten(case.base_output or '')} -> {_shorten(case.new_output or '')}"
        )
        lines.append(
            "    score:  "
            f"{fmt_score(case.base_score or 0)} -> {fmt_score(case.new_score or 0)} "
            f"(Δ {fmt_delta(case.score_delta or 0)})"
        )
    return lines


def _result_lines(report: Report) -> list[str]:
    if not report.summary.failed:
        return ["OK", "  within thresholds"]
    return ["FAIL", *(f"  {reason}" for reason in report.summary.fail_reasons)]


def _case_dict(case: CaseDelta, thresholds: Thresholds) -> dict:
    return {
        "id": case.id,
        "presence": case.presence,
        "prompt": case.prompt,
        "base_prompt": case.base_prompt,
        "new_prompt": case.new_prompt,
        "base_output": case.base_output,
        "new_output": case.new_output,
        "base_score": case.base_score,
        "new_score": case.new_score,
        "score_delta": case.score_delta,
        "base_latency_ms": case.base_latency_ms,
        "new_latency_ms": case.new_latency_ms,
        "latency_delta_ms": case.latency_delta_ms,
        "base_cost": case.base_cost,
        "new_cost": case.new_cost,
        "cost_delta": case.cost_delta,
        "score_regression": case.is_score_regression(thresholds),
        "newly_failing": case.is_newly_failing(thresholds),
        "improved": case.is_improved(thresholds),
        "still_failing": case.still_failing(thresholds),
        "note": _note(case, thresholds),
    }


def _note(case: CaseDelta, thresholds: Thresholds) -> str:
    notes: list[str] = []
    if case.is_score_regression(thresholds):
        notes.append("regression")
    if case.is_newly_failing(thresholds):
        notes.append("newly failing")
    if case.is_improved(thresholds):
        notes.append("improved")
    if case.still_failing(thresholds):
        notes.append("still failing")
    return ", ".join(notes)


def _sorted_by_drop(cases: Sequence[CaseDelta]) -> list[CaseDelta]:
    """Worst score change first. Ties use the rounded delta, then id.

    Rounding matches the four decimal places shown in the table, so two drops
    that both print as -0.2 stay in id order instead of binary-float order.
    """

    def key(case: CaseDelta) -> tuple[float, str]:
        delta = 0.0 if case.score_delta is None else round(case.score_delta, 4)
        return (delta, case.id)

    return sorted(cases, key=key)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    rendered = [[str(cell) for cell in row] for row in rows]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    body = [format_row(headers), format_row(["-" * width for width in widths])]
    body.extend(format_row(row) for row in rendered)
    return "\n".join(f"  {line}" if line else "  " for line in body)


def _delta_with_pct(delta: float, base: float, format_delta) -> str:
    if abs(base) <= 1e-9:
        return f"{format_delta(delta)} (from 0)"
    pct = delta / base * 100.0
    return f"{format_delta(delta)} ({fmt_pct(pct)})"


def _shorten(text: str, limit: int = _DETAIL_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def fmt_score(value: float) -> str:
    """Format a score without trailing zeros, keeping tiny non-zero values visible."""

    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    if text == "0" and value != 0:
        return f"{value:.6g}"
    return text


def fmt_delta(value: float) -> str:
    if value == 0:
        return "0"
    if value > 0:
        return f"+{fmt_score(value)}"
    return fmt_score(value)


def fmt_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def fmt_ms(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{round(value)} ms"
    return f"{value:.1f} ms"


def _signed_unit(value: float, format_value) -> str:
    if value == 0:
        return format_value(0)
    sign = "+" if value > 0 else "-"
    return sign + format_value(abs(value))


def fmt_signed_ms(value: float) -> str:
    return _signed_unit(value, fmt_ms)


def fmt_cost(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${_money(abs(value))}"


def fmt_signed_cost(value: float) -> str:
    if value == 0:
        return fmt_cost(0)
    sign = "+" if value > 0 else "-"
    return sign + fmt_cost(abs(value))


def _money(value: float) -> str:
    if value == 0:
        return "0.00"
    if value >= 0.0001:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.6g}"
    if "." not in text:
        text += ".00"
    elif len(text.split(".")[1]) == 1:
        text += "0"
    return text
