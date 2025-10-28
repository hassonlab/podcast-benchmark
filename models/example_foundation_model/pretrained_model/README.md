# Pretrained Model Directory

This directory represents a "pretrained foundation model" that can be loaded and used in the podcast benchmark framework.

## Contents

- **`config.yaml`**: Model architecture configuration
- **`checkpoint.pth`**: Model weights (randomly initialized for demonstration)
- **`generate_checkpoint.py`**: Script used to generate the checkpoint

## Current Status

⚠️ **Important**: This checkpoint contains **randomly initialized weights**, not pretrained weights!

## Usage

This model directory is loaded by the integration code in two ways:

### 1. Feature Extraction (Frozen)
```python
from example_foundation_model.simple_transformer import load_pretrained_model

# Load the model
model = load_pretrained_model("example_foundation_model/pretrained_model")
model.eval()
model.freeze()

# Extract features
with torch.no_grad():
    embeddings = model(neural_data)  # Returns [batch_size, 256]
```

### 2. Finetuning (Trainable)
```python
# The decoder class loads the foundation model and includes it as a submodule
decoder = FoundationModelDecoder(
    model_dir="example_foundation_model/pretrained_model",
    freeze_layers=2,  # Freeze first 2 layers, train the rest
)

# During training, gradients flow through unfrozen layers
output = decoder(neural_data)
```

## Regenerating the Checkpoint

If you modify `config.yaml` and want a new random checkpoint:

```bash
cd example_foundation_model/pretrained_model
conda activate decoding_env
python generate_checkpoint.py
```
