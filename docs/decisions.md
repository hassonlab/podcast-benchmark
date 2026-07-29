# Decisions

This document records why major repository structures exist so future changes can preserve the intent behind the codebase.

## Configurable Data -> Preprocessor -> Model -> Training Loop Path

The current setup is built around a configurable path from data loading, to preprocessing, to model construction, to the shared training loop.

The purpose is to support arbitrary decoding models on the same ECoG language benchmark without requiring new users to modify the underlying training foundation. Users can define a task target, register a preprocessor, register a model constructor, and describe the experiment in YAML. The shared runtime then handles data loading, lagged slicing, fold creation, training, checkpointing, metrics, and result writing.

This keeps the core benchmark surface controlled and comparable while still allowing new foundation models and neural decoders to plug in through registries. It also lets model-specific details live in `models/` and task-specific target construction live in `tasks/`, instead of spreading custom logic through the training loop.

## Registry-Based Extension Points

Datasets, models, tasks, preprocessors, metrics, config setters, and model data getters are registered through `core/registry.py`. This choice makes the codebase extensible without hard-coding every supported component into `main.py`.

## Canonical Stimulus Events and Subject Neural Clocks

Task tables describe canonical stimulus events and targets. A dataset may optionally map those rows to a different onset array for each subject. Lag slicing intersects rows that are valid for every selected Raw, then slices each Raw using its own neural-clock onsets. This supports shared-stimulus combined runs without resampling or warping recordings onto an artificial common clock.

Dataset getters return MNE Raw objects and mark dataset-specific bad channels in `raw.info['bads']`; shared loading then handles configured dropping, resampling, unit conversion, and channel selection. Dataset preprocessing that affects referencing, such as Brain Treebank corrupted-channel removal before common-average reference, is performed when the reusable FIF artifact is built.

Brain Treebank artifact construction resamples each HDF5 electrode while reading when a target rate is requested. Full-movie source recordings are too large to safely stack and copy at 2048 Hz before downsampling; channel-wise resampling preserves the artifact builder's output rate while bounding peak memory.

## Canonical Word Labels

Reusable word-level label artifacts use unique `event_id`, stimulus-time `start`, and optional `end`, `word`, and task columns. Linguistic task getters are views over that table. Legacy Podcast paths and column aliases remain accepted, while new datasets can generate one table for surprisal, content/function, POS, and sentence-onset tasks.

## Foundation Preprocessor Pipelines

Foundation model configs list their ordered preprocessing steps directly and do not wrap them with `disk_cache_preprocessor`. They keep `config_setter_name: foundation_feature_cache` so nested foundation model settings are still prepared before training. Model-specific steps such as BrainBERT/PopT STFT are declared as explicit preprocessor entries with their own params instead of hidden `DataParams` flags.

Nested foundation specs used by `foundation_feature_cache` are feature-extraction-only specs. They set `feature_cache: true` and omit task-specific head/training parameters such as `output_dim`, decoder activation, dropout, MLP sizes, and freeze controls. Legacy `embedding_dim` values are stripped when encountered. This keeps feature-cache identities stable across benchmark tasks within the same foundation model family; the downstream `mlp_probe_decoder` remains responsible for task-specific output shape and activation.

## Chunked Preprocessing For Large Runs

`DataParams.chunked_preprocessing` is an opt-in execution mode for lag datasets whose preprocessed tensors are too large to keep in memory. The default path still slices, preprocesses, and trains from a single in-memory tensor. When enabled, `RawNeuralDataset` computes the selected rows once per lag, preprocesses near-equal row chunks independently, writes temporary `.npz` files, and the training loop streams one chunk at a time through chunk-backed loaders.

The chunk files are temporary run artifacts, not persistent caches. They are deleted in a `finally` block after each lag so failures do not leave normal runs pinned to stale preprocessing output. Preprocessors that need whole-lag statistics should keep this mode disabled because each chunk is preprocessed independently.

Chunk files are written through temporary filenames and atomically moved into place only after a successful write, so partial `.npz` files from quota or I/O failures are tracked and removed. Cluster submissions through `submit.sh` override the chunk cache to a job-local temporary directory and install a shell cleanup trap, reducing shared scratch pressure when Slurm terminates a job before Python cleanup can finish.

DIVER volume-level lag sweeps can still create too much concurrent cache and scratch pressure when every lag batch is submitted independently. The dedicated Slurm helper submits those batches with `--dependency=singleton` and stable per-config job names, so each config advances through lag batches serially while different configs can still run in parallel.

DIVER volume-level configs use five temporary preprocessing chunks by default. This bounds peak memory while keeping each chunk large enough to avoid excessive disk-backed loader overhead; cluster submissions place these chunks in job-local temporary storage.

## Neural-Only Training Loop

The shared training loop no longer runs auxiliary sklearn/Himalaya baseline estimators or baseline-only mode. Baseline model families that are implemented as registered torch models, such as `neural_conv_decoder` or `linear_model`, remain regular model configs. This keeps fold training, metrics, checkpoints, and streaming chunk support on one neural model path.

## YAML Experiment Configuration

Experiments are primarily configured through YAML files in `configs/`. This makes benchmark runs reproducible, easy to generate in bulk, and comparable across models, tasks, subjects, regions, lags, folds, and training settings.

Paper-result consolidation may use explicit model/task lag bounds under `cleaning.lag_bounds`. Filtering is performed before coverage validation and runnable-config generation so a deliberately narrower analysis range does not require rewriting source result CSVs.
