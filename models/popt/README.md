# Example Foundation Model

A self-contained example demonstrating how to integrate a foundation model with the podcast benchmark framework.

## Overview

This example shows two ways to use a foundation model:

1. **Feature Extraction (Frozen)**: Extract embeddings during preprocessing, train a simple decoder
2. **Finetuning**: Include the foundation model in your decoder and train end-to-end

## Pattern 1: Feature Extraction

```python
@registry.register_data_preprocessor("example_foundation_feature_extraction")
def extract_foundation_features(data, preprocessor_params):
    model = load_pretrained_model(preprocessor_params["model_dir"])
    model.freeze()
    return model(data)  # Returns embeddings

@registry.register_model_constructor("example_foundation_mlp")
def create_mlp_decoder(model_params):
    return MLPDecoder(...)  # Train small MLP on frozen features
```

**When to use**: Fast experiments, good pretrained model, trying different decoders.

## Pattern 2: Finetuning

```python
class FoundationModelDecoder(nn.Module):
    def __init__(self, model_dir, num_frozen_layers=0, ...):
        self.foundation_model = load_pretrained_model(model_dir)
        self.foundation_model.freeze_layers(num_frozen_layers)  # Partial freezing
        self.decoder_head = MLPDecoder(...)

    def forward(self, x):
        features = self.foundation_model(x)
        return self.decoder_head(features)

@registry.register_model_constructor("example_foundation_finetune")
def create_finetuning_decoder(model_params):
    return FoundationModelDecoder(...)
```

**When to use**: Adapt the foundation model to your specific task, more expressive.

## Running the Examples

```bash
# Feature extraction
python main.py --config example_foundation_model/configs/feature_extraction.yaml

# Finetuning
python main.py --config example_foundation_model/configs/finetuning.yaml
```

## Key Files

- **`simple_transformer.py`**: Basic transformer with freeze/unfreeze methods
- **`integration.py`**: Both patterns fully implemented
- **`pretrained_model/`**: Example model directory structure

See `docs/onboarding-model.md` for detailed guide on adapting this for your own foundation model.
