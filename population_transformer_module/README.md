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

### Configuration Files

Three pre-configured experiments are available:

- `configs/population_transformer/population_transformer_base.yml`: Standard configuration
- `configs/population_transformer/population_transformer_frozen.yml`: Frozen PopulationTransformer weights
- `configs/population_transformer/population_transformer_finetune.yml`: End-to-end fine-tuning

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
└── population_transformer_finetune.yml
```

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

## Troubleshooting

1. **Import errors**: Ensure PopulationTransformer is properly installed and in your Python path
2. **Model loading issues**: Verify the model path and weights compatibility
3. **Memory issues**: Reduce batch size in preprocessing parameters
4. **Dimension mismatches**: Check `pt_embedding_dim` matches your model's output

## Next Steps

1. **Adapt position encoding** to match your specific electrode layout
2. **Optimize preprocessing** for your dataset's characteristics  
3. **Experiment with different** decoder architectures
4. **Add attention visualization** support if needed 