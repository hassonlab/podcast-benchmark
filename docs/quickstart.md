# Quickstart

Get started with the podcast benchmark framework in minutes.

## Setup

To download data and set up your local virtual environment:

```bash
./setup.sh
```

This will:
- Create a Python virtual environment (conda or venv)
- Install the core benchmark runtime dependencies
- Download the WordNet corpora required by word-embedding tasks into the environment
- Download the necessary podcast listening data

**Setup options**:
```bash
./setup.sh --gpu         # Install GPU dependencies (CUDA packages)
./setup.sh --dev         # Install dev dependencies (testing), skip data download
./setup.sh --docs        # Install documentation site dependencies
./setup.sh --paper       # Install paper result and atlas visualization dependencies
./setup.sh --audio       # Install audio/prosody transcription dependencies
./setup.sh --data        # Install dataset preprocessing dependencies
./setup.sh --diver-full  # Install vendored DIVER data loading dependencies
./setup.sh --all         # Install all optional dependencies
./setup.sh --env-name NAME # Custom environment name (default: decoding_env)
```

You can combine options, for example `./setup.sh --dev --paper`, to install
only the optional workflows you need.

## Interactive Tutorial

The [first-model marimo notebook](https://github.com/hassonlab/podcast-benchmark/blob/main/tutorials/getting_started.py) is the
smallest real-data introduction to the benchmark. Its default demo downloads
only subject 8's preprocessed high-gamma recording (about 265 MB); the
sentence-onset labels are already included in the repository.

```bash
pip install -e ".[tutorial]"
marimo edit tutorials/getting_started.py
```

The notebook defines and registers a temporal preprocessor and linear decoder,
runs a two-fold sentence-onset experiment at three lags with one
`run_benchmark(config)` call, and plots held-out ROC-AUC. A second section
constructs and registers a four-electrode synthetic MNE Raw dataset, then reuses
the same model, preprocessor, and task with only configuration changes.

A generated [Jupyter notebook](https://github.com/hassonlab/podcast-benchmark/blob/main/tutorials/getting_started.ipynb)
contains the same tutorial in sequential form and can be
[opened directly in Colab](https://colab.research.google.com/github/hassonlab/podcast-benchmark/blob/main/tutorials/getting_started.ipynb).
Run all cells, then save or commit the notebook to preserve its tables and plots.
Regenerate it after editing the marimo source with:

```bash
python scripts/export_tutorial_ipynb.py
```

## Training Your First Model

The framework comes with several pre-configured models you can train immediately.

### 1. Neural Convolutional Decoder

This recreates the decoder from [Goldstein et al. 2022](https://www.nature.com/articles/s41593-022-01026-4), which decodes word embeddings directly from neural data:

```bash
make neural-conv
```

### 2. Foundation Model Decoder

This trains a decoder from a foundation model's latent representations to word embeddings:

```bash
make foundation-model
```

### 3. Foundation Models

Evaluate one of the pre-configured foundation models on word embedding
decoding:

```bash
python main.py --config configs/foundation_models/popt/word_embedding/supersubject.yml
```

Available production foundation model configs include:

- **BrainBERT**: `configs/foundation_models/brainbert/<task>/<variant>.yml`
- **DIVER**: `configs/foundation_models/diver/<task>/<variant>.yml`
- **POPT**: `configs/foundation_models/popt/<task>/<variant>.yml`

For example:

```bash
python main.py --config configs/foundation_models/brainbert/word_embedding/supersubject.yml
python main.py --config configs/foundation_models/diver/word_embedding/supersubject.yml
python main.py --config configs/foundation_models/popt/word_embedding/supersubject.yml
```

Each foundation model currently follows the same task layout, including
`word_embedding`, `whisper_embedding`, `llm_embedding_pretraining`,
`llm_decoding`, `sentence_onset`, `gpt_surprise`,
`gpt_surprise_multiclass`, `content_noncontent`, `pos`, `iu_boundary`, and
`volume_level`.

## Results

Training results will be saved to:
- `results/` - Performance metrics and CSV files
- `checkpoints/` - Saved model checkpoints
- `event_logs/` - TensorBoard logs

See [Baseline Results](baseline-results.md) for performance benchmarks across all tasks.

## Configuration

To modify data, behavior, or hyperparameters:

Edit the relevant configuration file in `configs/`:
- `configs/baselines/<task_name>/<baseline_family>/` - Task-grouped baseline configs, including supersubject, per-subject, and per-region variants
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
