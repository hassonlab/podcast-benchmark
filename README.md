# Podcast Benchmark

A benchmarking framework for neural decoding from podcast listening data.

Comparing brain → word decoding performance to [previously published results](https://www.nature.com/articles/s41593-022-01026-4).

## Documentation

**📚 Full documentation available at: https://hassonlab.github.io/podcast-benchmark/**

## Quick Start

```bash
# Setup environment and download data
./setup.sh

# Train neural convolutional decoder
make neural-conv

# Train foundation model decoder
make foundation-model
```

## Features

- **Flexible model architecture**: Register custom models with simple decorators
- **Multiple tasks**: Word embeddings, classification, or custom prediction targets
- **Configurable training**: YAML-based configs with cross-validation and early stopping
- **Multiple metrics**: ROC-AUC, perplexity, top-k accuracy, and custom metrics
- **Time lag analysis**: Automatically find optimal temporal offsets

## Learn More

- **[Quickstart Guide](https://hassonlab.github.io/podcast-benchmark/quickstart/)** - Get up and running
- **[Onboarding a Model](https://hassonlab.github.io/podcast-benchmark/onboarding-model/)** - Add your own models
- **[Adding a Task](https://hassonlab.github.io/podcast-benchmark/adding-task/)** - Create custom tasks
- **[Configuration](https://hassonlab.github.io/podcast-benchmark/configuration/)** - Understanding configs
- **[Registry API](https://hassonlab.github.io/podcast-benchmark/api-reference/)** - Function signatures
