# PopulationTransformer Integration with podcast-benchmark

This module integrates the PopulationTransformer model into the podcast-benchmark framework for neural decoding experiments.

## Overview

The PopulationTransformer integration allows you to:
1. Use pre-trained PopulationTransformer models to extract neural embeddings
2. Decode word embeddings from these neural representations
3. Compare PopulationTransformer performance against other baselines (neural_conv_decoder, foundation_model)

## Components

### Model Architecture
- **PopulationTransformerDecoder**: A simple MLP that decodes word embeddings from PopulationTransformer neural embeddings
- Follows the same pattern as the foundation_model approach but uses PopulationTransformer features

### Data Preprocessing
- **population_transformer_preprocessing_fn**: Extracts features from neural data using a pre-trained PopulationTransformer model
- Handles batching, device management, and position encoding
- Outputs neural embeddings that can be decoded to word embeddings

### Configuration System
- **population_transformer_config_setter**: Dynamically sets model parameters at runtime
- Sets input/output dimensions based on data characteristics
- Handles electrode mapping and PopulationTransformer-specific parameters

## PopulationTransformer Architecture Alignment

The PopulationTransformer integration in podcast-benchmark follows the standard architecture described in the literature and HuggingFace documentation. Here's how each component maps to our code and setup:

### 1. Frozen Temporal Encoder
- **Purpose:** Extracts per-channel temporal embeddings from raw neural signals.
- **Examples:** BrainBERT, TOTEM, TS2Vec, Chronos.
- **Status in Our Integration:**
  - The preprocessing function (`population_transformer_preprocessing_fn`) loads a pre-trained temporal encoder (e.g., BrainBERT) and extracts embeddings for each channel. The encoder weights are frozen during inference.
  - **Output:** For each channel, a temporal embedding vector of dimension \( d \).

### 2. Population Transformer (PopT) Encoder
- **Purpose:** Applies a full transformer stack to temporally encoded signals, adding spatial context and attention.
- **Architecture in Our Integration:**
  - **Layers:** 6
  - **Attention Heads:** 8
  - **Hidden Dimension:** 512
  - **Input:** Temporally encoded signals + positional embeddings
  - **Output:**
    - One embedding per channel
    - One [CLS] token summarizing the population-level brain state
- **Status in Our Integration:**
  - The core PopulationTransformer model is loaded and used at inference time, as specified in the YAML config files.

### 3. [CLS] Token + Task-Specific Head
- **Purpose:** The [CLS] token summarizes the ensemble-level brain state and is used for downstream tasks.
- **Implementation in Our Integration:**
  - The [CLS] token is extracted from the transformer output and passed through a simple linear (MLP) head for the decoding task (e.g., word embedding prediction).
  - This is implemented in the `PopulationTransformerDecoder` (MLP head).
  - **Output:** The output of the linear head is used for the downstream task (e.g., word embedding regression, classification, etc.).

### Alignment Table

| Paper/Description Component         | Our Integration (Code/Config)                |
|-------------------------------------|----------------------------------------------|
| Frozen Temporal Encoder             | Preprocessing function, frozen model weights |
| Population Transformer (6L, 8H, 512)| `pt_model` config, transformer stack         |
| [CLS] Token + Linear Head           | `PopulationTransformerDecoder` (MLP head)    |

## Usage

### Running Experiments

1. **Base configuration** (recommended starting point):
   ```bash
   make population-transformer-base
   ```

2. **Frozen PopulationTransformer** (only train decoder head):
   ```bash
   make population-transformer-frozen
   ```

3. **Fine-tuning** (end-to-end training):
   ```bash
   make population-transformer-finetune
   ```

4. **CPU-optimized** (for testing without GPU):
   ```bash
   make population-transformer-cpu
   ```

### Configuration Files

Four pre-configured experiments are available:

- `configs/population_transformer/population_transformer_base.yml`: Standard configuration
- `configs/population_transformer/population_transformer_frozen.yml`: Frozen PopulationTransformer weights
- `configs/population_transformer/population_transformer_finetune.yml`: End-to-end fine-tuning
- `configs/population_transformer/population_transformer_cpu.yml`: CPU-optimized for testing

### Key Parameters

#### Model Parameters
- `input_dim`: PopulationTransformer embedding dimension (default: 512)
- `output_dim`: Word embedding dimension (default: 50)
- `hidden_dims`: MLP decoder layer sizes (default: [256, 128])

#### Preprocessing Parameters
- `model_path`: Path to pre-trained PopulationTransformer weights
- `batch_size`: Batch size for PopulationTransformer inference
- `use_cls_token`: Whether to use CLS token or mean pooling
- `frozen_weights`: Whether to freeze PopulationTransformer during training
- `pt_embedding_dim`: Expected PopulationTransformer output dimension

## Prerequisites

1. **PopulationTransformer weights**: Download pre-trained weights from the [PopulationTransformer repository](https://github.com/czlwang/PopulationTransformer) or [HuggingFace](https://huggingface.co/PopulationTransformer/popt_brainbert_stft)

2. **Data format**: Ensure your neural data is compatible with PopulationTransformer's expected input format

3. **Dependencies**: Install both podcast-benchmark and PopulationTransformer dependencies

## Integration Architecture

The integration follows podcast-benchmark's modular design:

```
population_transformer_module/
├── __init__.py
├── population_transformer_utils.py  # Core integration code
└── README.md                        # This file

configs/population_transformer/
├── population_transformer_base.yml
├── population_transformer_frozen.yml
├── population_transformer_finetune.yml
└── population_transformer_cpu.yml
```

## Important Changes Made During Integration

### 1. Environment Setup
- **Virtual Environment**: Created `decoding_env` virtual environment for isolated dependencies
- **Dependencies**: Installed core packages: `torch`, `numpy`, `mne`, `mne-bids`, `gensim`, `h5py`, `tensorboard`, `pyyaml`
- **CPU Support**: Configured for CPU-only execution (no CUDA required)

### 2. Import Path Fixes
- **PopulationTransformer Imports**: Fixed import paths in `population_transformer_utils.py` to use `population_transformer.models` and `population_transformer.preprocessors`
- **Foundation Model**: Implemented lazy imports for `ecog_foundation_model` to avoid dependency conflicts when running PopulationTransformer experiments

### 3. Configuration Updates
- **Model Paths**: Updated all config files to use correct path: `population_transformer/pretrained_weights/popt_brainbert_stft/pretrained_popt_brainbert_stft.pth`
- **CPU Configuration**: Created CPU-optimized config with smaller batch sizes and reduced complexity
- **Device Handling**: Added proper device management for CPU/CUDA compatibility

### 4. Registry Integration
- **Model Constructor**: Registered `population_transformer_mlp` for MLP decoder
- **Data Preprocessor**: Registered `population_transformer_preprocessing_fn` for neural data processing
- **Config Setter**: Registered `population_transformer_config_setter` for runtime parameter setting

### 5. Makefile Integration
- **New Targets**: Added `population-transformer`, `population-transformer-frozen`, `population-transformer-finetune`, `population-transformer-base`, `population-transformer-cpu`
- **Command Structure**: Follows existing pattern for consistency

## Customization

To customize the integration:

1. **Modify the decoder architecture** in `PopulationTransformerDecoder`
2. **Adjust preprocessing** in `population_transformer_preprocessing_fn`
3. **Create new configs** following the existing pattern
4. **Update position encoding** in `create_dummy_positions` for your electrode layout

## Results and Comparison

Results will be saved in the `results/` directory with the trial name. You can compare:
- PopulationTransformer vs. neural_conv_decoder
- PopulationTransformer vs. foundation_model  
- Different PopulationTransformer configurations (frozen vs. fine-tuned)

## Current Core Issue: Data Loading Problem

The current issue is that the `.fif` files in `data/derivatives/ecogprep` are corrupted or incomplete (only ~214 bytes or 1KB). This suggests that the `setup.sh` script's download of these files was likely interrupted or failed.

### **Symptoms:**
- `ValueError: file does not start with a file id tag` when loading `.fif` files
- Small file sizes (~1KB) instead of expected large neural data files
- Data loading fails during preprocessing

### **Root Cause:**
The neural data files were not properly downloaded during the initial setup process.

### **Solution:**
Re-download the neural data files by:
1. Deleting the corrupted `ds005574` directory
2. Redownload manually
3. Ensuring complete download of all neural data files




