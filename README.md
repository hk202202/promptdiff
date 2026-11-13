# promptdiff

promptdiff compares two LLM eval runs and reports what got worse. Each run is a JSONL file with one scored case per line. The report lists per-case score deltas, cases that newly fail, and latency and cost changes. For CI, the process exits 1 when a regression count or a latency or cost increase is greater than the threshold you set.

Python 3.9 or newer. The runtime has no third-party dependencies.

## Install

From this repository:

```bash
pip install -e .
```

Check the entry point:

```bash
promptdiff --help
```

`python -m promptdiff` runs the same command.

## Eval files

One JSON object per line:

```json
{"id": "translate", "prompt": "Translate 'hello' to Spanish.", "output": "hola", "score": 1.0, "latency_ms": 180, "cost": 0.0015}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string or integer | Case id. Compared as text, so `7` and `"7"` are the same case. |
| `prompt` | string | Input sent to the model. |
| `output` | string | Model output. |
| `score` | number | Higher is better. |
| `latency_ms` | number | End-to-end latency. Must be `>= 0`. |
| `cost` | number | Cost of the call. Must be `>= 0`. |

`id` and `score` drive the join and the pass/fail gate. The other fields are required so a partial export fails loudly instead of looking like a free latency win. Blank lines are skipped. Duplicate ids are an error. Extra keys are ignored.

A case passes when its score is greater than or equal to `--pass-score` (default `0.5`). A shared case is a score regression when the baseline score minus the new score is greater than `--min-score-drop` (default `0`). A newly failing case passed in the baseline and fails in the candidate. Those two sets overlap when a drop is large enough to cross the pass score, and CI can gate them separately.

Score, latency, and cost aggregates use cases that appear in both files. Added and removed cases are listed on their own. Removed cases fail the run only when you set `--max-removed`.

Differences of `1e-9` or less are treated as ties, so binary float noise does not become a regression. A count or an increase equal to its limit still passes. The run fails when the value is greater than the limit.

## Usage

```bash
promptdiff base.jsonl new.jsonl
promptdiff base.jsonl new.jsonl --min-score-drop 0.05 --max-regressions 2
promptdiff base.jsonl new.jsonl --max-latency-increase 20 --max-cost-increase 15
promptdiff base.jsonl new.jsonl --format json
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--pass-score` | `0.5` | Score at or above this is a pass. |
| `--min-score-drop` | `0` | Minimum score decrease that counts as a regression. |
| `--max-regressions` | `0` | Fail when score regressions exceed this count. |
| `--max-new-failures` | `0` | Fail when newly failing cases exceed this count. |
| `--max-removed` | off | Fail when removed cases exceed this count. |
| `--max-latency-increase` | off | Fail when mean latency rises by more than this percent. |
| `--max-cost-increase` | off | Fail when total cost rises by more than this percent. |
| `--format` | `text` | `text` or `json`. JSON keeps baseline order and includes `summary.failed`. |
| `--show-all` | off | Also list latency and cost decreases. |
| `-v`, `--verbose` | off | Print the prompt and output for regressions and new failures. |

Percent flags take a number such as `25` for 25%. An increase from zero fails any finite percent gate, because the percent change is unbounded.

Latency and cost tables list increases only, largest absolute change first.

## Demo

`examples/base.jsonl` and `examples/new.jsonl` are a small before-and-after:

- `translate` answers in French and drops from 1 to 0, so it newly fails.
- `summarize` and `cite-year` score lower. `cite-year` was already below the pass score.
- `json-extract` improves. `math` and `qa-capital` hold their scores. `math` gets slower and more expensive.
- `dropped-case` is only in the baseline. `added-case` is only in the candidate.

```bash
promptdiff examples/base.jsonl examples/new.jsonl
```

The report below is the current output. The process exits 1:

```text
base  examples/base.jsonl
new   examples/new.jsonl

Summary
  shared             6
  added / removed    1 / 1
  pass score         0.5
  mean score         0.8167 -> 0.6167 (Δ -0.2)
  score regressions  3 (max 0, drop > 0)
  newly failing      1 (max 0)
  improved           1
  unchanged          2
  mean latency       208.3 ms -> 278.3 ms (+33.6%, 6 cases)
  total cost         $0.0125 -> $0.0173 (+38.4%, 6 cases)

Score deltas
  id            base  new  delta  note
  ------------  ----  ---  -----  -------------------------
  translate     1     0    -1     regression, newly failing
  cite-year     0.2   0    -0.2   regression, still failing
  summarize     0.9   0.7  -0.2   regression
  math          1     1    0
  qa-capital    1     1    0
  json-extract  0.8   1    +0.2   improved

Newly failing
  id         score
  ---------  ------
  translate  1 -> 0

Case coverage
  removed
    dropped-case
  added
    added-case

Latency increases
  id         base    new     delta
  ---------  ------  ------  -----------------
  math       200 ms  450 ms  +250 ms (+125.0%)
  summarize  340 ms  520 ms  +180 ms (+52.9%)
  cite-year  150 ms  160 ms  +10 ms (+6.7%)
  translate  180 ms  190 ms  +10 ms (+5.6%)

Cost increases
  id         base     new      delta
  ---------  -------  -------  ------------------
  summarize  $0.004   $0.0065  +$0.0025 (+62.5%)
  math       $0.0018  $0.0042  +$0.0024 (+133.3%)
  cite-year  $0.001   $0.0011  +$0.0001 (+10.0%)
  translate  $0.0015  $0.0016  +$0.0001 (+6.7%)

FAIL
  3 score regressions exceed max 0
  1 newly failing case exceeds max 0
```

The same report exits 0 when the counts are inside the limits:

```bash
promptdiff examples/base.jsonl examples/new.jsonl --max-regressions 3 --max-new-failures 1
```

A CI job that also budgets latency and cost:

```bash
promptdiff eval/base.jsonl eval/candidate.jsonl \
  --max-regressions 0 \
  --max-new-failures 0 \
  --max-latency-increase 25 \
  --max-cost-increase 15
```

`--format json` prints the same facts as one object with `thresholds`, `summary` (`failed`, `fail_reasons`, means, totals), and `cases`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Every gated count and increase is within its limit. |
| 1 | A regression count or an increase exceeded its limit. |
| 2 | A file or a flag is invalid. |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
