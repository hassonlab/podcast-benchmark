# Repository Overview

This repository provides a decoding suite for decoding language information from ECoG data collected during podcast listening. It supports configurable benchmark tasks, neural decoding models, and several foundation model integrations under a shared training and evaluation interface.

## Main Flow

`main.py` is the experiment entry point. It loads YAML configuration, imports registered models, tasks, metrics, and preprocessors, loads ECoG data, builds task targets, applies preprocessing, and runs training over configured lags and folds.

The main pipeline is:

1. A config in `configs/` defines the task, data parameters, model, training settings, and output paths.
2. A task in `tasks/` returns a DataFrame with event times and prediction targets.
3. `utils/data_utils.py` loads subject ECoG data from `data/`.
4. `utils/dataset.py` slices lagged neural windows and applies registered preprocessors. Large runs can opt into disk-backed preprocessing chunks with `data_params.chunked_preprocessing.enabled`.
5. `utils/model_utils.py` builds the configured model from the registry.
6. `utils/decoding_utils.py` trains and evaluates the model across folds and lags.
7. Metrics from `metrics/` are written to result directories.

## Important Folders and Files

`main.py` contains the top-level single-task and multi-task execution logic, including run modes for combined, per-subject, and per-region experiments.

`core/registry.py` defines the decorator registries for model constructors, task data getters, preprocessors, config setters, metrics, and model-specific data getters. This is the main extension surface.

`core/config.py` defines the dataclasses used by YAML configs, including `ExperimentConfig`, `TaskConfig`, `DataParams`, `TrainingParams`, and `ModelSpec`.

`configs/` contains experiment YAML files. `configs/foundation_models/` holds generated benchmark configs for foundation models and tasks. These configs list foundation preprocessing steps directly; BrainBERT and PopT configs declare STFT as an explicit step before `foundation_feature_cache`, and DIVER configs call `foundation_feature_cache` directly. Nested foundation specs inside `foundation_feature_cache` are cache-only feature extraction specs and intentionally omit task head parameters. Volume-level foundation configs opt into temporary chunked preprocessing so large lag tensors are not fully materialized after preprocessing. `configs/examples/` contains smaller examples.

`tasks/` defines decoding targets such as word embeddings, Whisper embeddings, sentence onset, content/non-content words, part of speech, LLM surprise, IU boundaries, volume level, and LLM decoding. Each task registers a data getter and task-specific config.

`models/` contains model implementations and integrations. Foundation models such as `brainbert/`, `diver/`, and `popt/` live here alongside compact neural model families such as `neural_conv_decoder/`, `linear_model/`, and `time_pooling_model/`. Shared helpers, decoders, config setters, and preprocessors are in the top-level `models/*.py` files. `models/shared_preprocessors.py` includes a `disk_cache_preprocessor` wrapper that caches the final output of one registered preprocessor or an ordered preprocessor pipeline under `.cache/preprocessors/` by default. Its `update_cache_with_missing` option defaults to `false`; when enabled, missing rows computed during partial reuse are merged back into the same canonical cache file.

Model configs use `model_params.output_dim` for decoder output size across regression, classification, and embedding targets.

`metrics/` contains registered losses and evaluation metrics for regression, classification, embedding prediction, and language model decoding.

`utils/` contains shared runtime utilities: data loading, lagged dataset construction, fold creation, model construction, config parsing, training, plotting, atlas/region helpers, and module auto-loading.

`data/` contains the podcast ECoG dataset and preprocessing code. Runtime data loading expects BIDS-style paths under this directory.

`processed_data/` contains derived CSV and text files used by tasks, such as word-level annotations, sentence data, electrode selections, and region metadata.

`benchmark-results/` and `paper-results/` contain generated benchmark outputs and paper-oriented result artifacts.

`scripts/` contains one-off and batch utilities for generating configs, training targets, paper results, scoring, profiling, transcription, and analysis.

`tests/` contains pytest coverage for configs, registries, data loading, datasets, tasks, metrics, model integrations, scripts, and smoke tests for the training loop.

`docs/` contains user-facing documentation, including quickstart, model onboarding, task creation, configuration, task reference, baseline results, API reference, and these repository notes.

`README.md` gives a concise project summary and links to the hosted documentation.

`pyproject.toml` declares package metadata and dependency groups for runtime, tests, docs, paper results, audio tooling, data preprocessing, and full GPU support.

`mkdocs.yml` configures the documentation site built from `docs/`.

`Makefile`, `setup.sh`, and `submit.sh` provide setup and execution conveniences for local or cluster workflows. `submit_diver_volume_lags.sh` submits DIVER volume-level lag batches with Slurm singleton dependencies so each config runs one lag batch at a time.

`training_matrix.yaml` describes benchmark training coverage across models, tasks, and run modes.
