# Decisions

This document records why major repository structures exist so future changes can preserve the intent behind the codebase.

## Configurable Data -> Preprocessor -> Model -> Training Loop Path

The current setup is built around a configurable path from data loading, to preprocessing, to model construction, to the shared training loop.

The purpose is to support arbitrary decoding models on the same ECoG language benchmark without requiring new users to modify the underlying training foundation. Users can define a task target, register a preprocessor, register a model constructor, and describe the experiment in YAML. The shared runtime then handles data loading, lagged slicing, fold creation, training, checkpointing, metrics, and result writing.

This keeps the core benchmark surface controlled and comparable while still allowing new foundation models and simple baselines to plug in through registries. It also lets model-specific details live in `models/` and task-specific target construction live in `tasks/`, instead of spreading custom logic through the training loop.

## Registry-Based Extension Points

Models, tasks, preprocessors, metrics, config setters, and model data getters are registered through `core/registry.py`. This choice makes the codebase extensible without hard-coding every supported model or task into `main.py`.

## Foundation Preprocessor Pipelines

Foundation model configs list their preprocessing steps directly and recompute preprocessing on each run. They keep `config_setter_name: foundation_feature_cache` so nested foundation model settings are still prepared before training, but they do not wrap the pipeline with the registered `disk_cache_preprocessor`. Model-specific steps such as BrainBERT/PopT STFT are declared as explicit preprocessor entries with their own params instead of hidden `DataParams` flags. This keeps checked-in configs simple and avoids reusing stale on-disk preprocessing outputs.

## YAML Experiment Configuration

Experiments are primarily configured through YAML files in `configs/`. This makes benchmark runs reproducible, easy to generate in bulk, and comparable across models, tasks, subjects, regions, lags, folds, and training settings.
