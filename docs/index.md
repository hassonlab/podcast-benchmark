# Podcast Benchmark Documentation

A benchmarking framework for neural decoding from podcast listening data.

Comparing brain → word decoding performance to [previously published results](https://www.nature.com/articles/s41593-022-01026-4).

## Table of Contents

1. [Quickstart](quickstart.md) - Get up and running quickly
2. [Onboarding a New Model](onboarding-model.md) - Step-by-step guide to adding your own decoding model
3. [Adding a New Task](adding-task.md) - How to implement custom decoding tasks
4. [Configuration Guide](configuration.md) - Understanding and configuring experiments
5. [Registry API Reference](api-reference.md) - Registry decorators and function signatures

## Overview

This framework provides a flexible system for:
- Training neural decoding models on iEEG data
- Comparing different model architectures
- Evaluating performance across multiple metrics
- Running systematic hyperparameter searches

For long updates and discussions, see [this notebook](https://docs.google.com/document/d/1IE1v_CyjZxTYaYVncxctJqZYzmYyFIgdZLXpKvEMaqc/edit?usp=sharing).
