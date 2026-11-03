# Podcast Benchmark

A benchmarking framework for neural decoding from podcast listening data.

## Documentation

**📚 Full documentation available at: https://hassonlab.github.io/podcast-benchmark/**

## Quick Start

New to the repository? Start with the
**[interactive marimo tutorial](tutorials/getting_started.py)**. It can download a
single-subject demo and walks through registering a linear model, running the
sentence-onset task, plotting performance across lags, and registering a new
synthetic MNE dataset without changing the model or task.

Prefer Jupyter or Google Colab? Open the generated
**[Jupyter notebook](tutorials/getting_started.ipynb)** or
**[run it in Colab](https://colab.research.google.com/github/hassonlab/podcast-benchmark/blob/main/tutorials/getting_started.ipynb)**.

```bash
# Setup environment and download data
./setup.sh

# Train all tasks using the CNN baseline.
make train-all MODEL=baselines/neural_conv_decoder
```

## Features

- **Flexible model architecture**: Register custom models with simple decorators
- **Multiple tasks**: Word embeddings, classification, or custom prediction targets
- **Extensible datasets**: Register MNE Raw loaders and subject-specific event clocks
- **Configurable training**: YAML-based configs with cross-validation and early stopping
- **Multiple metrics**: ROC-AUC, perplexity, top-k accuracy, and custom metrics
- **Time lag analysis**: Automatically find optimal temporal offsets

## Learn More

- **[Quickstart Guide](https://hassonlab.github.io/podcast-benchmark/quickstart/)** - Get up and running
- **[Marimo Tutorial](tutorials/getting_started.py)** - Train a first model on the one-subject demo
- **[Jupyter/Colab Tutorial](tutorials/getting_started.ipynb)** - Sequential notebook ready to run and save with outputs
- **[Onboarding a Model](https://hassonlab.github.io/podcast-benchmark/onboarding-model/)** - Add your own models
- **[Adding a Task](https://hassonlab.github.io/podcast-benchmark/adding-task/)** - Create custom tasks
- **[Adding a Dataset](https://hassonlab.github.io/podcast-benchmark/adding-dataset/)** - Add neural datasets
- **[Configuration](https://hassonlab.github.io/podcast-benchmark/configuration/)** - Understanding configs
- **[Registry API](https://hassonlab.github.io/podcast-benchmark/api-reference/)** - Function signatures
