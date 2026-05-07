# Insights

Post-hoc analysis scripts that read the test reports written under
`pynguin_report/` and produce summaries, plots, or statistical tests.

None of these are required to run experiments -- they only consume results.

## Files

- **`coverage_diff.py`** -- Compares average branch coverage between two
  configurations (e.g. DEEPMOSA vs CODAMOSA) module-by-module, listing where
  one beats the other.
- **`metrics_table.py`** -- Aggregates metrics (coverage, LLM calls, tokens,
  query time, etc.) across all projects/configs and prints a comparison table.
- **`coverage_plot.py`** -- Plots branch-coverage timelines per config using
  matplotlib/seaborn.
- **`win_tie_loss.py`** -- Computes Win/Tie/Loss counts between configs with
  Mann-Whitney U significance tests and Vargha-Delaney A12 effect size.
- **`rerun_tests.py`** -- Uses coverage.py to compare per-function coverage
  between the algorithm's final suite and the raw LLM tests, quantifying
  coverage lost during Pynguin's deserialization.
- **`count_types.py`** -- Walks each project's AST with `astroid` and counts
  type-annotation usage across modules.

## Running

Each script is a module under the `experiments.insights` package. Run from the
repo root:

```bash
python -m experiments.insights.metrics_table
python -m experiments.insights.coverage_plot
# etc.
```
