import numpy as np
import os
import torch

from tqdm import tqdm

from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
from ecog_foundation_model.config import (
    create_video_mae_experiment_config_from_yaml,
    VideoMAEExperimentConfig,
    ECoGDataConfig,
)
from ecog_foundation_model.ecog_setup import create_model, CheckpointManager
from ecog_foundation_model import model_registry

from config import ExperimentConfig, dict_to_config
import registry


def create_foundation_model(config: VideoMAEExperimentConfig, model_dir=None):
    model = create_model(config)
    if model_dir:
        ckpt_manager = CheckpointManager(model)
        ckpt_manager.load(os.path.join(model_dir, "checkpoint.pth"), strict=False)
    return model


def prepare_data_for_foundation_model(
    data, data_config: ECoGDataConfig, ch_names: list[str]
):
    data = data.reshape(
        data.shape[0], data.shape[1], -1, data_config.original_fs // data_config.new_fs
    )
    data = data.mean(-1)

    for i in range(64):
        channel = "G" + str(i + 1)
        if not np.isin(channel, ch_names):
            data = np.insert(data, i, np.zeros_like(data[:, i, :]) + np.nan, axis=1)

    # Reshape to [num_examples, frequency bands (currrently 1), time, num_electrodes]
    data = np.einsum("bet->bte", data).reshape(data.shape[0], data.shape[2], 8, 8)
    data = np.expand_dims(data, axis=1)

    return data


@registry.register_data_preprocessor("foundation_model_finetune_mlp")
def prepare_data_for_finetuning(data, preprocessor_params):
    data_config = preprocessor_params["ecog_data_config"]
    return prepare_data_for_foundation_model(
        data, data_config, preprocessor_params["ch_names"]
    )


@registry.register_config_setter("foundation_model")
def foundation_model_config_setter(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    preprocessor_params = (
        experiment_config.data_params.preprocessor_params
        if experiment_config.data_params.preprocessor_params
        else {}
    )
    preprocessor_params["ch_names"] = ch_names

    # Set window width to whatever the sample length of the foundation model is.
    ecog_config = create_video_mae_experiment_config_from_yaml(
        os.path.join(preprocessor_params["model_dir"], "experiment_config.yml")
    )
    experiment_config.data_params.window_width = (
        ecog_config.ecog_data_config.sample_length
    )
    experiment_config.model_params["model_dim"] = (
        ecog_config.video_mae_task_config.vit_config.dim
    )
    experiment_config.model_params["model_name"] = (
        ecog_config.video_mae_task_config.model_name
    )
    return experiment_config


@registry.register_data_preprocessor()
def foundation_model_preprocessing_fn(data, preprocessor_params):
    ecog_config = create_video_mae_experiment_config_from_yaml(
        os.path.join(preprocessor_params["model_dir"], "experiment_config.yml")
    )
    model = create_foundation_model(ecog_config, preprocessor_params["model_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data = prepare_data_for_foundation_model(
        data, ecog_config.ecog_data_config, preprocessor_params["ch_names"]
    )

    # Construct input dataset
    batch_size = preprocessor_params["foundation_model_batch_size"]
    foundation_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = torch.tensor(data[i : i + batch_size], dtype=torch.float32).to(
                device
            )
            batch_embeddings = model(
                batch,
                forward_features=True,
                global_pool=preprocessor_params["global_pool"],
            )  # Shape: [batch_size, 16]
            foundation_embeddings.append(batch_embeddings.cpu().numpy())

    foundation_embeddings = np.vstack(foundation_embeddings)

    return foundation_embeddings


def _initialize_foundation_model_config(experiment_config):
    """Loads and sets full foundation model config as VideoMAEExperimentConfig."""

    def _set_dropout_rates(experiment_config, foundation_config):
        if experiment_config.model_params.get("proj_drop"):
            foundation_config.video_mae_task_config.vit_config.proj_drop = (
                experiment_config.model_params["proj_drop"]
            )
        if experiment_config.model_params.get("drop_path"):
            foundation_config.video_mae_task_config.vit_config.drop_path = (
                experiment_config.model_params["drop_path"]
            )

    if experiment_config.model_params.get("model_dir"):
        config_path = os.path.join(
            experiment_config.model_params["model_dir"], "experiment_config.yml"
        )
        foundation_config = create_video_mae_experiment_config_from_yaml(config_path)
        # Override dropout rates if provided.
        _set_dropout_rates(experiment_config, foundation_config)
        experiment_config.model_params["foundation_model_config"] = foundation_config
        experiment_config.model_params["model_name"] = (
            foundation_config.video_mae_task_config.model_name
        )
    else:
        foundation_config = dict_to_config(
            experiment_config.model_params["foundation_model_config"],
            VideoMAEExperimentConfig,
        )
        if experiment_config.model_params.get("model_name"):
            foundation_config.video_mae_task_config.vit_config = (
                model_registry.model_registry[
                    experiment_config.model_params["model_name"]
                ]
            )
        # Override dropout rates if provided.
        _set_dropout_rates(experiment_config, foundation_config)
        experiment_config.model_params["foundation_model_config"] = foundation_config

    # Set top-level data window width
    experiment_config.data_params.window_width = (
        foundation_config.ecog_data_config.sample_length
    )

    # Copy ECoG-specific preprocessor params
    experiment_config.data_params.preprocessor_params["ecog_data_config"] = (
        foundation_config.ecog_data_config
    )

    return foundation_config


@registry.register_config_setter("foundation_model_finetune_mlp")
def foundation_model_mlp_finetune_config_setter(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    experiment_config.data_params.preprocessor_params = {}

    # Add channel names
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    experiment_config.data_params.preprocessor_params["ch_names"] = ch_names

    # Load and resolve foundation model config
    foundation_config = _initialize_foundation_model_config(experiment_config)

    encoder_dim = foundation_config.video_mae_task_config.vit_config.dim
    print("Encoder Dim:", encoder_dim)
    experiment_config.model_params["model_dim"] = encoder_dim

    return experiment_config


@registry.register_config_setter("foundation_model_finetune_conv")
def foundation_model_conv_finetune_config_setter(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    experiment_config.data_params.preprocessor_params = {}

    # Include list of channel names
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    experiment_config.data_params.preprocessor_params["ch_names"] = ch_names

    # Load and resolve foundation model config
    foundation_config = _initialize_foundation_model_config(experiment_config)

    # Get relevant fields
    vit_cfg = foundation_config.video_mae_task_config.vit_config
    ecog_cfg = foundation_config.ecog_data_config

    # Compute spatiotemporal grid shape: (T, H, W)
    T = ecog_cfg.sample_length * ecog_cfg.new_fs // vit_cfg.frame_patch_size
    H = 8 // vit_cfg.patch_size
    W = 8 // vit_cfg.patch_size
    experiment_config.model_params["grid_shape"] = [T, H, W]

    # Optionally fill in default for conv_out_dim
    if "conv_out_dim" not in experiment_config.model_params:
        experiment_config.model_params["conv_out_dim"] = (
            experiment_config.data_params.embedding_pca_dim
        )

    return experiment_config
