# Decisions

This document records why major repository structures exist so future changes can preserve the intent behind the codebase.

## Configurable Data -> Preprocessor -> Model -> Training Loop Path

The current setup is built around a configurable path from data loading, to preprocessing, to model construction, to the shared training loop.

The purpose is to support arbitrary decoding models on the same ECoG language benchmark without requiring new users to modify the underlying training foundation. Users can define a task target, register a preprocessor, register a model constructor, and describe the experiment in YAML. The shared runtime then handles data loading, lagged slicing, fold creation, training, checkpointing, metrics, and result writing.

This keeps the core benchmark surface controlled and comparable while still allowing new foundation models and neural decoders to plug in through registries. It also lets model-specific details live in `models/` and task-specific target construction live in `tasks/`, instead of spreading custom logic through the training loop.

## Registry-Based Extension Points

Models, tasks, preprocessors, metrics, config setters, and model data getters are registered through `core/registry.py`. This choice makes the codebase extensible without hard-coding every supported model or task into `main.py`.

## Foundation Preprocessor Pipelines

Foundation model configs list their ordered preprocessing steps directly and do not wrap them with `disk_cache_preprocessor`. They keep `config_setter_name: foundation_feature_cache` so nested foundation model settings are still prepared before training. Model-specific steps such as BrainBERT/PopT STFT are declared as explicit preprocessor entries with their own params instead of hidden `DataParams` flags.

Nested foundation specs used by `foundation_feature_cache` are feature-extraction-only specs. They set `feature_cache: true` and omit task-specific head/training parameters such as `output_dim`, decoder activation, dropout, MLP sizes, and freeze controls. Legacy `embedding_dim` values are stripped when encountered. This keeps feature-cache identities stable across benchmark tasks within the same foundation model family; the downstream `mlp_probe_decoder` remains responsible for task-specific output shape and activation.

## Chunked Preprocessing For Large Runs

`DataParams.chunked_preprocessing` is an opt-in execution mode for lag datasets whose preprocessed tensors are too large to keep in memory. The default path still slices, preprocesses, and trains from a single in-memory tensor. When enabled, `RawNeuralDataset` computes the selected rows once per lag, preprocesses near-equal row chunks independently, writes temporary `.npz` files, and the training loop streams one chunk at a time through chunk-backed loaders.

The chunk files are temporary run artifacts, not persistent caches. They are deleted in a `finally` block after each lag so failures do not leave normal runs pinned to stale preprocessing output. Preprocessors that need whole-lag statistics should keep this mode disabled because each chunk is preprocessed independently.

## Neural-Only Training Loop

The shared training loop no longer runs auxiliary sklearn/Himalaya baseline estimators or baseline-only mode. Baseline model families that are implemented as registered torch models, such as `neural_conv_decoder` or `linear_model`, remain regular model configs. This keeps fold training, metrics, checkpoints, and streaming chunk support on one neural model path.

## YAML Experiment Configuration

Experiments are primarily configured through YAML files in `configs/`. This makes benchmark runs reproducible, easy to generate in bulk, and comparable across models, tasks, subjects, regions, lags, folds, and training settings.
