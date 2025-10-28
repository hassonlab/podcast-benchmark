"""
Script to generate a randomly initialized checkpoint for demonstration purposes.

This creates a checkpoint file that demonstrates the structure of a saved model,
but with random weights. In a real scenario, this would be replaced with weights
from actual pretraining.
"""

import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from simple_transformer import create_model_from_config


def generate_checkpoint(config_path: str, output_path: str):
    """
    Generate a randomly initialized checkpoint.

    Args:
        config_path: Path to config.yaml file
        output_path: Path where checkpoint should be saved
    """
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print("Creating model...")
    model = create_model_from_config(config)

    print(f"Model has {model.get_num_params():,} parameters")

    # Save checkpoint with metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "note": "This is a randomly initialized checkpoint for demonstration purposes only.",
        "config": {
            "input_channels": config.input_channels,
            "model_dim": config.model_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
        }
    }

    print(f"Saving checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    output_path = os.path.join(script_dir, "checkpoint.pth")

    generate_checkpoint(config_path, output_path)
