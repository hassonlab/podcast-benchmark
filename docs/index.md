# Podcast Benchmark Documentation

A benchmarking framework for neural decoding from podcast listening data. 

## Decoding Tasks

1. **Brain --> perceived word decoding** Translate brain signals to perceived words, comparing performance to [previously published results](https://www.nature.com/articles/s41593-022-01026-4).
2. **Audio Reconstruction** Reconstruct podcast audio envelope from brain signal (Regression)
3. **Sentence Onset Detection** Classify (binary) segments of brain data as containing the beginning of a sentence or not
4. **Content/Non-Content Words Classification** (Binary classification)
5. **Part of Speech Classification** (Multiclass classification)
6. **LLM Surprise** Predict how likely the perceived word is given it's context (Regression)
7. **LLM Decoding** Encode brain data as vector input to language models (GPT-2) for direct brain-to-text generation

## Table of Contents

1. [Quickstart](quickstart.md) - Get up and running quickly
2. [Onboarding a New Model](onboarding-model.md) - Step-by-step guide to adding your own decoding model
3. [Adding a New Task](adding-task.md) - How to implement custom decoding tasks
4. [Configuration Guide](configuration.md) - Understanding and configuring experiments
5. [Task Reference](task-reference.md) - Complete reference for all available tasks
6. [Baseline Results](baseline-results.md) - Performance benchmarks for all tasks
7. [Registry API Reference](api-reference.md) - Registry decorators and function signatures

## Overview

This framework provides a flexible system for:
- Training neural decoding models on iEEG data
- Comparing different model architectures
- Evaluating performance across multiple metrics
- Running systematic hyperparameter searches

For long updates and discussions, see [this notebook](https://docs.google.com/document/d/1IE1v_CyjZxTYaYVncxctJqZYzmYyFIgdZLXpKvEMaqc/edit?usp=sharing).
