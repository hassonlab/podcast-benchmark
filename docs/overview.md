# Repository Overview

This repository provides a decoding suite for decoding language information from ECoG data collected during podcast listening. It supports configurable benchmark tasks, neural decoding models, and several foundation model integrations under a shared training and evaluation interface.

## Main Flow

`main.py` is the experiment entry point. It loads YAML configuration, imports registered models, tasks, metrics, and preprocessors, loads ECoG data, builds task targets, applies preprocessing, and runs training over configured lags and folds.

The main pipeline is:

1. A config in `configs/` defines the dataset, task, data parameters, model, training settings, and output paths.
2. A task in `tasks/` returns a canonical DataFrame with event times and prediction targets.
3. A registered getter in `datasets/` loads each subject's MNE Raw and optionally maps canonical stimulus times to that subject's neural clock.
4. `utils/dataset.py` slices lagged neural windows and applies registered preprocessors. Large runs can opt into disk-backed preprocessing chunks with `data_params.chunked_preprocessing.enabled`.
5. `utils/model_utils.py` builds the configured model from the registry.
6. `utils/decoding_utils.py` trains and evaluates the model across folds and lags. Single-lag null controls can set `training_params.num_null_repetitions` to produce independently seeded repetitions, per-repetition metrics/checkpoints, and an aggregate null summary.
7. Metrics from `metrics/` are written to result directories. Runs with `training_params.save_test_predictions: true` also write best-checkpoint, out-of-fold test outputs to `test_predictions.h5` for paired analyses; full LLM token-logit outputs are intentionally unsupported.

## Important Folders and Files

`main.py` contains the top-level single-task and multi-task execution logic, including run modes for combined, per-subject, and per-region experiments.

`core/registry.py` defines the decorator registries for datasets, model constructors, task data getters, preprocessors, config setters, metrics, and model-specific data getters. This is the main extension surface.

`core/config.py` defines the dataclasses used by YAML configs, including `ExperimentConfig`, `TaskConfig`, `DataParams`, `TrainingParams`, and `ModelSpec`. A model spec can set `random_init: true` to load the requested architecture/checkpoint and then replace every learned parameter with a fresh random initialization. Native module initialization hooks are preferred; unchanged custom parameters receive a normal-distribution fallback controlled by PyTorch's random seed.

`configs/` contains experiment YAML files. `configs/brain_treebank/` contains matching per-subject suites for the linear model, CNN, BrainBERT, DIVER, and PopT across canonical Brain Treebank word-label tasks. Brain Treebank foundation YAMLs each contain one subject and use 20 disk-backed preprocessing chunks per lag; they can be regenerated with `scripts/generate_brain_treebank_foundation_configs.py`. `scripts/generate_brain_treebank_coordinates.py` converts the public localization table into the per-subject MNI sidecars required by DIVER under `processed_data/brain_treebank/coordinates`, which can remain writable when the raw-data mount is read-only. `configs/foundation_models/` holds Podcast benchmark configs for foundation models and tasks. These configs list foundation preprocessing steps directly; BrainBERT and PopT configs declare STFT as an explicit step before `foundation_feature_cache`, and DIVER configs call `foundation_feature_cache` directly. Nested foundation specs inside `foundation_feature_cache` are cache-only feature extraction specs and intentionally omit task head parameters. `configs/controls/foundation_random_init/` and `configs/controls/foundation_shuffled_targets/` mirror all foundation-model super-subject and individual-subject tasks. Their generators create 10-repetition random-init and 100-repetition shuffled-target controls, respectively, with a single default lag that submission overrides per job. Volume-level foundation configs can opt into temporary chunked preprocessing for large lag tensors; DIVER volume-level configs use five chunks to bound peak memory. `configs/examples/` contains smaller examples.

`tasks/` defines decoding targets such as word embeddings, Whisper embeddings, sentence onset, content/non-content words, part of speech, LLM surprise, IU boundaries, volume level, and LLM decoding. Each task registers a data getter and task-specific config.

`datasets/` contains registered MNE Raw getters. Podcast remains the default; Brain Treebank loads precomputed per-subject/per-movie broadband or high-gamma FIF artifacts and maps movie-time task rows through subject sync tables.

`models/` contains model implementations and integrations. Foundation models such as `brainbert/`, `diver/`, and `popt/` live here alongside compact neural model families such as `neural_conv_decoder/`, `linear_model/`, and `time_pooling_model/`. Shared helpers, decoders, config setters, and preprocessors are in the top-level `models/*.py` files. `models/shared_preprocessors.py` includes a `disk_cache_preprocessor` wrapper that caches the final output of one registered preprocessor or an ordered preprocessor pipeline under `.cache/preprocessors/` by default. Its `update_cache_with_missing` option defaults to `false`; when enabled, missing rows computed during partial reuse are merged back into the same canonical cache file.

Model configs use `model_params.output_dim` for decoder output size across regression, classification, and embedding targets.

`metrics/` contains registered losses and evaluation metrics for regression, classification, embedding prediction, and language model decoding.

`utils/` contains shared runtime utilities: data loading, lagged dataset construction, fold creation, model construction, config parsing, training, plotting, atlas/region helpers, and module auto-loading.
`utils/raw_preprocessing.py` contains reusable MNE despiking, broadband cleaning, and high-gamma helpers. `utils/label_utils.py` normalizes canonical word-label tables and legacy Podcast column names.

`data/` contains the podcast ECoG dataset and preprocessing code. Runtime data loading expects BIDS-style paths under this directory.

`processed_data/` contains derived CSV and text files used by tasks, such as word-level annotations, sentence data, electrode selections, and region metadata.

`benchmark-results/` and `paper-results/` contain generated benchmark outputs and paper-oriented result artifacts.

`scripts/` contains one-off and batch utilities for generating configs, training targets, paper results, scoring, profiling, transcription, and analysis.
`scripts/significance_test.py` runs YAML-configured paired block-permutation tests over saved out-of-fold prediction artifacts. It supports best-lag comparisons and per-lag comparisons against a baseline, reuses registered scalar metrics, and writes Holm-corrected result tables with the resolved analysis config.
`scripts/preprocess_brain_treebank.py` converts Brain Treebank HDF5 trials into benchmark FIF and sync artifacts, resampling each source channel during loading to keep full-movie conversion memory bounded. `scripts/build_word_labels.py` creates canonical Podcast or Brain Treebank word-label tables. Small Brain Treebank CNN proof-of-concept configs live under `configs/examples/brain_treebank/`.
`scripts/clean_paper_result_config.py` can apply model/task-specific inclusive lag bounds from `cleaning.lag_bounds` while consolidating result shards.

`tests/` contains pytest coverage for configs, registries, data loading, datasets, tasks, metrics, model integrations, scripts, and smoke tests for the training loop.

`docs/` contains user-facing documentation, including quickstart, model onboarding, task creation, configuration, task reference, baseline results, API reference, and these repository notes.

`README.md` gives a concise project summary and links to the hosted documentation.

`pyproject.toml` declares package metadata and dependency groups for runtime, tests, docs, paper results, audio tooling, data preprocessing, and full GPU support.

`mkdocs.yml` configures the documentation site built from `docs/`.

`Makefile`, `setup.sh`, and `submit.sh` provide setup and execution conveniences for local or cluster workflows. `setup.sh` installs the WordNet corpora used by word-embedding tasks into the selected Python environment so network-isolated batch jobs can load them without a runtime download. `submit.sh` gives chunked preprocessing a job-local temporary cache directory and removes it on shell exit. `submit_diver_volume_lags.sh` submits DIVER volume-level lag batches with Slurm singleton dependencies so each config runs one lag batch at a time. The `train-foundation-random-init-controls` and `train-foundation-shuffled-target-controls` targets submit separate jobs at -500, 0, and 500 ms for matching foundation benchmark conditions; `FOUNDATION_CONTROL_SCOPE` can select `all`, `super_subject`, or `per_subject`, and stable job names serialize the three lags for each condition.
`train-foundation-one-shot-controls` submits the generated random-init and shuffled-target configs as ordinary one-repetition lag sweeps for the significant-electrode supersubject and subjects 1-9. It excludes LLM decoding and splits the -1000 to 1000 ms sweep into two non-overlapping jobs per condition.
The `fill-paper-result-gaps` Make target runs the exact remaining baseline gap jobs indexed by `benchmark-results/results_results.yml`; its dry-run mode prints every job without executing it.

`training_matrix.yaml` describes benchmark training coverage across models, tasks, and run modes.
