import torch
import argparse
import os
import subprocess

from ecog_foundation_model.ecog_setup import CheckpointManager, create_model
from ecog_foundation_model.config import create_video_mae_experiment_config_from_yaml


def convert_checkpoint(input_path, output_path, config=None, optimizer=None):
    ecog_config = create_video_mae_experiment_config_from_yaml(
        os.path.join(input_path, "experiment_config.yml")
    )
    checkpoint = torch.load(
        os.path.join(input_path, "model.pth"),
        weights_only=True,
        map_location="cpu",
    )
    model = create_model(ecog_config)
    model.load_state_dict(checkpoint, strict=False)
    model.initialize_mask(None)

    checkpoint_manager = CheckpointManager(model=model)

    checkpoint_manager.save(output_path)
    print(f"[✔] Converted checkpoint saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to original checkpoint (e.g., model.pth)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save new formatted checkpoint"
    )
    args = parser.parse_args()

    # Optional: import your actual config class here
    # from foundation_model.config import ModelConfig
    # config = ModelConfig()

    # Optional: create an optimizer if you want to include it
    # model = FoundationModel(config)
    # optimizer = torch.optim.Adam(model.parameters())

    # For now just skip config/optimizer unless needed
    convert_checkpoint(args.input, args.output, config=None, optimizer=None)
