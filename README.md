# Podcast Benchmark

A benchmarking framework for neural decoding from podcast listening data.

## Documentation

**📚 Full documentation available at: https://hassonlab.github.io/podcast-benchmark/**

## Quick Start

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
- **[Onboarding a Model](https://hassonlab.github.io/podcast-benchmark/onboarding-model/)** - Add your own models
- **[Adding a Task](https://hassonlab.github.io/podcast-benchmark/adding-task/)** - Create custom tasks
- **[Adding a Dataset](https://hassonlab.github.io/podcast-benchmark/adding-dataset/)** - Add neural datasets
- **[Configuration](https://hassonlab.github.io/podcast-benchmark/configuration/)** - Understanding configs
- **[Registry API](https://hassonlab.github.io/podcast-benchmark/api-reference/)** - Function signatures
