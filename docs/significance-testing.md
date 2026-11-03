# Prediction Significance Testing

Runs created with `training_params.save_test_predictions: true` can be compared
with paired block-permutation tests. Each input must be a single run-unit directory
containing `test_predictions.h5`; parent directories containing per-subject or
per-region subruns are not accepted.

Create an analysis YAML and pass it to `scripts/significance_test.py`:

```yaml
mode: best_lag
metric:
  name: corr
  higher_is_better: true
results:
  - name: model_a
    path: ../results/model_a
  - name: model_b
    path: ../results/model_b
valid_lags: null
block_size: 1
n_permutations: 10000
random_seed: 42
alpha: 0.05
output_dir: ../significance/model_a_vs_model_b
```

Paths are resolved relative to the analysis YAML. The metric name must identify a
registered scalar metric. Scores are recomputed over pooled out-of-fold predictions,
so they can differ from the mean of fold metrics in `lag_performance.csv`.

Run the analysis with:

```bash
python scripts/significance_test.py path/to/analysis.yml
```

The output directory contains the resolved `config.yml` and `p_values.csv`. All
p-values are one-sided tests that the first named result is better in the configured
metric direction, and `p_value_holm` controls familywise error across the complete
analysis family.

## Modes

### Best lag

`best_lag` selects candidate lags from the intersection available across all results,
then optionally restricts that set with `valid_lags`. All results and lags are scored
on the same common samples. Every permutation applies the same event-block swap mask
at all lags and reselects each result's best lag before comparing scores.

The observed overall winner is reported against every other result in `p_values.csv`.
Because that winner is data-selected, Holm correction is calculated over every
ordered result pair. The complete correction family is saved to
`all_pairwise_tests.csv`.

### Baseline by lag

Use `baseline_lags` to compare each result with a named baseline at every shared lag:

```yaml
mode: baseline_lags
metric:
  name: mse
  higher_is_better: false
baseline:
  name: baseline
  path: ../results/baseline
results:
  - name: model_a
    path: ../results/model_a
valid_lags: null
block_size: 10
n_permutations: 10000
random_seed: 42
alpha: 0.05
output_dir: ../significance/model_a_vs_baseline
```

Holm correction covers every result-by-lag hypothesis in the output.

## Statistical assumptions

Predictions and targets are joined by `sample_id`. At a given lag, input results must
have exactly the same IDs, targets, onsets, and prediction shapes. The test swaps the
two results' predictions in contiguous onset-ordered blocks; it never shuffles targets
or prediction rows. `block_size` is an event count and should be large enough that
separate blocks can reasonably be treated as exchangeable under the null. A block size
of one is appropriate only when event-level differences are effectively independent.

The test evaluates the sharp null that the paired prediction blocks are exchangeable
between results. It does not by itself establish equivalence when a comparison is not
significant.
