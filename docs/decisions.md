# Decisions

This document records why major repository structures exist so future changes can preserve the intent behind the codebase.

## Configurable Data -> Preprocessor -> Model -> Training Loop Path

The current setup is built around a configurable path from data loading, to preprocessing, to model construction, to the shared training loop.

The purpose is to support arbitrary decoding models on the same ECoG language benchmark without requiring new users to modify the underlying training foundation. Users can define a task target, register a preprocessor, register a model constructor, and describe the experiment in YAML. The shared runtime then handles data loading, lagged slicing, fold creation, training, checkpointing, metrics, and result writing.

This keeps the core benchmark surface controlled and comparable while still allowing new foundation models and simple baselines to plug in through registries. It also lets model-specific details live in `models/` and task-specific target construction live in `tasks/`, instead of spreading custom logic through the training loop.

## Registry-Based Extension Points

Models, tasks, preprocessors, metrics, config setters, and model data getters are registered through `core/registry.py`. This choice makes the codebase extensible without hard-coding every supported model or task into `main.py`.

## Disk-Cached Preprocessor Pipelines

Expensive preprocessing can be wrapped with the registered `disk_cache_preprocessor`. The cache stores the final output of a full ordered preprocessor pipeline, not intermediate stages, and keys entries by wrapped preprocessor names, implementation source hashes, normalized parameters, lag, selected event starts, ordered subject electrode names, subject channel counts, and cache mode. This keeps repeated foundation feature extraction reproducible while invalidating automatically when implementation, parameters, selected rows, electrodes, lag, or mode changes.

## YAML Experiment Configuration

Experiments are primarily configured through YAML files in `configs/`. This makes benchmark runs reproducible, easy to generate in bulk, and comparable across models, tasks, subjects, regions, lags, folds, and training settings.
