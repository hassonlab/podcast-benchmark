# Quickstart

Get started with the podcast benchmark framework in minutes.

## Setup

To download data and set up your local virtual environment:

```bash
./setup.sh
```

This will:
- Create a Python virtual environment (conda or venv)
- Install all required dependencies
- Download the necessary podcast listening data

**Setup options**:
```bash
./setup.sh --gpu           # Install GPU dependencies (CUDA packages)
./setup.sh --dev           # Install dev dependencies (testing), skip data download
./setup.sh --env-name NAME # Custom environment name (default: decoding_env)
```

## Training Your First Model

The framework comes with several pre-configured models you can train immediately.

### 1. Neural Convolutional Decoder

This recreates the decoder from [Tang et al. 2022](https://www.nature.com/articles/s41593-022-01026-4), which decodes word embeddings directly from neural data:

```bash
make neural-conv
```

### 2. Foundation Model Decoder

This trains a decoder from a foundation model's latent representations to word embeddings:

```bash
make foundation-model
```

### 3. POPT Foundation Model

Evaluate the POPT foundation model on word embedding decoding:

```bash
python main.py --config configs/foundation_models/popt/word_embedding/supersubject.yml
```

## Results

Training results will be saved to:
- `results/` - Performance metrics and CSV files
- `checkpoints/` - Saved model checkpoints
- `event_logs/` - TensorBoard logs

See [Baseline Results](baseline-results.md) for performance benchmarks across all tasks.

## Configuration

To modify data, behavior, or hyperparameters:

Edit the relevant configuration file in `configs/`:
- `configs/baselines/neural_conv_decoder/` - Neural convolutional decoder baselines
- `configs/baselines/time_pooling_model/` - Time-pooling regression baselines
- `configs/examples/example_foundation_model/` - Example foundation-model configs
- `configs/foundation_models/` - Production foundation-model configs
- `configs/controls/llm_decoding/` - Control runs for LLM decoding
- `configs/hpo/` - Hyperparameter search grids

Model implementations can be found in the `models/` directory.

See [Onboarding a New Model](onboarding-model.md) for details on configuration options.

## Next Steps

- [Add your own model](onboarding-model.md)
- [Create a custom task](adding-task.md)
- [View all available tasks](task-reference.md)
- [Compare against baseline results](baseline-results.md)
- [Explore the API](api-reference.md)
