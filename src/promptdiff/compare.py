"""Join two eval runs and decide whether the candidate regressed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from promptdiff.errors import PromptdiffError
from promptdiff.load import EvalCase, load_jsonl

# Absorbs binary float noise so a drop of 0.2 does not count as more than 0.2.
_EPS = 1e-9


def exceeds(value: float, limit: float) -> bool:
    """True when value is greater than limit by more than float noise."""

    return value - limit > _EPS


@dataclass(frozen=True)
class Thresholds:
    """CI gates. Counts fail when they are greater than the max, not equal to it."""

    pass_score: float = 0.5
    min_score_drop: float = 0.0
    max_regressions: int = 0
    max_new_failures: int = 0
    max_removed: int | None = None
    max_latency_increase_pct: float | None = None
    max_cost_increase_pct: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.pass_score):
            raise PromptdiffError("pass score must be finite")
        if not math.isfinite(self.min_score_drop) or self.min_score_drop < 0:
            raise PromptdiffError("min score drop must be >= 0")
        if self.max_regressions < 0 or self.max_new_failures < 0:
            raise PromptdiffError("regression limits must be >= 0")
        if self.max_removed is not None and self.max_removed < 0:
            raise PromptdiffError("max removed must be >= 0")
        for name, value in (
            ("max latency increase", self.max_latency_increase_pct),
            ("max cost increase", self.max_cost_increase_pct),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise PromptdiffError(f"{name} must be >= 0")


@dataclass(frozen=True)
class CaseDelta:
    """One id across the baseline and the candidate."""

    id: str
    base_prompt: str | None
    new_prompt: str | None
    base_output: str | None
    new_output: str | None
    base_score: float | None
    new_score: float | None
    base_latency_ms: float | None
    new_latency_ms: float | None
    base_cost: float | None
    new_cost: float | None

    @property
    def prompt(self) -> str:
        if self.base_prompt is not None:
            return self.base_prompt
        return self.new_prompt or ""

    @property
    def presence(self) -> str:
        if self.base_score is None:
            return "added"
        if self.new_score is None:
            return "removed"
        return "shared"

    @property
    def score_delta(self) -> float | None:
        if self.base_score is None or self.new_score is None:
            return None
        return self.new_score - self.base_score

    @property
    def score_drop(self) -> float | None:
        delta = self.score_delta
        if delta is None:
            return None
        return -delta

    @property
    def latency_delta_ms(self) -> float | None:
        if self.base_latency_ms is None or self.new_latency_ms is None:
            return None
        return self.new_latency_ms - self.base_latency_ms

    @property
    def cost_delta(self) -> float | None:
        if self.base_cost is None or self.new_cost is None:
            return None
        return self.new_cost - self.base_cost

    def is_score_regression(self, thresholds: Thresholds) -> bool:
        drop = self.score_drop
        return drop is not None and exceeds(drop, thresholds.min_score_drop)

    def is_newly_failing(self, thresholds: Thresholds) -> bool:
        if self.base_score is None or self.new_score is None:
            return False
        return _passes(self.base_score, thresholds.pass_score) and _fails(
            self.new_score, thresholds.pass_score
        )

    def is_improved(self, thresholds: Thresholds) -> bool:
        delta = self.score_delta
        return delta is not None and exceeds(delta, thresholds.min_score_drop)

    def still_failing(self, thresholds: Thresholds) -> bool:
        if self.base_score is None or self.new_score is None:
            return False
        return _fails(self.base_score, thresholds.pass_score) and _fails(
            self.new_score, thresholds.pass_score
        )


@dataclass(frozen=True)
class Summary:
    """Aggregate counts and the reasons the run failed, if any."""

    shared: int
    added: int
    removed: int
    mean_score_base: float | None
    mean_score_new: float | None
    mean_score_delta: float | None
    regressions: int
    new_failures: int
    improved: int
    unchanged: int
    mean_latency_base_ms: float | None
    mean_latency_new_ms: float | None
    mean_latency_increase_pct: float | None
    mean_latency_from_zero: bool
    latency_pairs: int
    total_cost_base: float | None
    total_cost_new: float | None
    total_cost_increase_pct: float | None
    total_cost_from_zero: bool
    cost_pairs: int
    failed: bool
    fail_reasons: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    """Comparison of two eval files."""

    base_path: str
    new_path: str
    thresholds: Thresholds
    cases: tuple[CaseDelta, ...]
    summary: Summary


def compare_paths(
    base_path: str | Path,
    new_path: str | Path,
    thresholds: Thresholds | None = None,
) -> Report:
    """Load two JSONL files and compare them."""

    base = load_jsonl(base_path)
    new = load_jsonl(new_path)
    return compare_cases(
        base,
        new,
        thresholds,
        base_path=str(base_path),
        new_path=str(new_path),
    )


def compare_cases(
    base: dict[str, EvalCase],
    new: dict[str, EvalCase],
    thresholds: Thresholds | None = None,
    *,
    base_path: str = "base",
    new_path: str = "new",
) -> Report:
    """Compare two id maps. Order is the baseline order, then added ids."""

    gates = thresholds if thresholds is not None else Thresholds()
    ordered: list[str] = list(base)
    for case_id in new:
        if case_id not in base:
            ordered.append(case_id)
    cases = tuple(_delta(case_id, base.get(case_id), new.get(case_id)) for case_id in ordered)
    return Report(
        base_path=base_path,
        new_path=new_path,
        thresholds=gates,
        cases=cases,
        summary=_summarize(cases, gates),
    )


def _delta(case_id: str, base: EvalCase | None, new: EvalCase | None) -> CaseDelta:
    return CaseDelta(
        id=case_id,
        base_prompt=None if base is None else base.prompt,
        new_prompt=None if new is None else new.prompt,
        base_output=None if base is None else base.output,
        new_output=None if new is None else new.output,
        base_score=None if base is None else base.score,
        new_score=None if new is None else new.score,
        base_latency_ms=None if base is None else base.latency_ms,
        new_latency_ms=None if new is None else new.latency_ms,
        base_cost=None if base is None else base.cost,
        new_cost=None if new is None else new.cost,
    )


def _summarize(cases: tuple[CaseDelta, ...], thresholds: Thresholds) -> Summary:
    shared = [case for case in cases if case.presence == "shared"]
    added = sum(1 for case in cases if case.presence == "added")
    removed = sum(1 for case in cases if case.presence == "removed")
    regressions = [case for case in shared if case.is_score_regression(thresholds)]
    new_failures = [case for case in shared if case.is_newly_failing(thresholds)]
    improved = [case for case in shared if case.is_improved(thresholds)]
    unchanged = [
        case
        for case in shared
        if not case.is_score_regression(thresholds) and not case.is_improved(thresholds)
    ]
    base_scores = [case.base_score for case in shared if case.base_score is not None]
    new_scores = [case.new_score for case in shared if case.new_score is not None]
    mean_base = _mean(base_scores)
    mean_new = _mean(new_scores)
    mean_delta = None if mean_base is None or mean_new is None else mean_new - mean_base

    latency_base = [case.base_latency_ms for case in shared if _has_latency(case)]
    latency_new = [case.new_latency_ms for case in shared if _has_latency(case)]
    lat_base_mean = _mean(latency_base)
    lat_new_mean = _mean(latency_new)
    lat_pct, lat_from_zero = _paired_increase(lat_base_mean, lat_new_mean)

    cost_base_values = [case.base_cost for case in shared if _has_cost(case)]
    cost_new_values = [case.new_cost for case in shared if _has_cost(case)]
    cost_base = None if not cost_base_values else sum(cost_base_values)
    cost_new = None if not cost_new_values else sum(cost_new_values)
    cost_pct, cost_from_zero = _paired_increase(cost_base, cost_new)

    reasons: list[str] = []
    regression_reason = _count_reason(
        len(regressions),
        thresholds.max_regressions,
        "score regression",
        "score regressions",
    )
    if regression_reason:
        reasons.append(regression_reason)
    failure_reason = _count_reason(
        len(new_failures),
        thresholds.max_new_failures,
        "newly failing case",
        "newly failing cases",
    )
    if failure_reason:
        reasons.append(failure_reason)
    if thresholds.max_removed is not None:
        removed_reason = _count_reason(
            removed, thresholds.max_removed, "removed case", "removed cases"
        )
        if removed_reason:
            reasons.append(removed_reason)
    latency_reason = _increase_reason(
        "mean latency",
        lat_pct,
        lat_from_zero,
        lat_base_mean,
        lat_new_mean,
        thresholds.max_latency_increase_pct,
    )
    if latency_reason:
        reasons.append(latency_reason)
    cost_reason = _increase_reason(
        "total cost",
        cost_pct,
        cost_from_zero,
        cost_base,
        cost_new,
        thresholds.max_cost_increase_pct,
    )
    if cost_reason:
        reasons.append(cost_reason)

    return Summary(
        shared=len(shared),
        added=added,
        removed=removed,
        mean_score_base=mean_base,
        mean_score_new=mean_new,
        mean_score_delta=mean_delta,
        regressions=len(regressions),
        new_failures=len(new_failures),
        improved=len(improved),
        unchanged=len(unchanged),
        mean_latency_base_ms=lat_base_mean,
        mean_latency_new_ms=lat_new_mean,
        mean_latency_increase_pct=lat_pct,
        mean_latency_from_zero=lat_from_zero,
        latency_pairs=len(latency_base),
        total_cost_base=cost_base,
        total_cost_new=cost_new,
        total_cost_increase_pct=cost_pct,
        total_cost_from_zero=cost_from_zero,
        cost_pairs=len(cost_base_values),
        failed=bool(reasons),
        fail_reasons=tuple(reasons),
    )


def _has_latency(case: CaseDelta) -> bool:
    return case.base_latency_ms is not None and case.new_latency_ms is not None


def _has_cost(case: CaseDelta) -> bool:
    return case.base_cost is not None and case.new_cost is not None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _paired_increase(base: float | None, new: float | None) -> tuple[float | None, bool]:
    if base is None or new is None:
        return None, False
    if abs(base) <= _EPS:
        if exceeds(new, base):
            return None, True
        return 0.0, False
    return (new - base) / base * 100.0, False


def _passes(score: float, pass_score: float) -> bool:
    return not _fails(score, pass_score)


def _fails(score: float, pass_score: float) -> bool:
    return exceeds(pass_score, score)


def _count_reason(count: int, limit: int, singular: str, plural: str) -> str | None:
    if not exceeds(float(count), float(limit)):
        return None
    noun = singular if count == 1 else plural
    verb = "exceeds" if count == 1 else "exceed"
    return f"{count} {noun} {verb} max {limit}"


def _increase_reason(
    label: str,
    pct: float | None,
    from_zero: bool,
    base: float | None,
    new: float | None,
    limit: float | None,
) -> str | None:
    if limit is None or base is None or new is None or not exceeds(new, base):
        return None
    if from_zero:
        return f"{label} increased from 0 (max {_format_limit(limit)}%)"
    if pct is not None and exceeds(pct, limit):
        return f"{label} increased {pct:.2f}% (max {_format_limit(limit)}%)"
    return None


def _format_limit(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"
