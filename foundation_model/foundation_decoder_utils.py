import numpy as np
import os
import torch

from tqdm import tqdm

from foundation_model.model_code.models_mae import MaskedAutoencoderViT
from foundation_model.model_code.config import (
    create_video_mae_experiment_config_from_file,
    VideoMAEExperimentConfig,
)
from foundation_model.model_code.utils import create_model

from config import ExperimentConfig, dict_to_config
import registry


@registry.register_config_setter("foundation_model")
def foundation_model_config_setter(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    preprocessor_params = experiment_config.data_params.preprocessor_params
    preprocessor_params["ch_names"] = ch_names

    # Set window width to whatever the sample length of the foundation model is.
    ecog_config = create_video_mae_experiment_config_from_file(
        os.path.join(preprocessor_params["model_dir"], "experiment_config.ini")
    )
    experiment_config.data_params.window_width = (
        ecog_config.ecog_data_config.sample_length
    )
    return experiment_config


@registry.register_data_preprocessor()
def foundation_model_preprocessing_fn(data, preprocessor_params):
    # First load the model and set it in eval mode.
    ecog_config = create_video_mae_experiment_config_from_file(
        os.path.join(preprocessor_params["model_dir"], "experiment_config.ini")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        os.path.join(preprocessor_params["model_dir"], "model_weight_test.pth"),
        weights_only=True,
    )
    model = create_model(ecog_config)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()

    data_config = ecog_config.ecog_data_config
    data = data.reshape(
        data.shape[0], data.shape[1], -1, data_config.original_fs // data_config.new_fs
    )
    data = data.mean(-1)

    for i in range(64):
        channel = "G" + str(i + 1)
        if not np.isin(channel, preprocessor_params["ch_names"]):
            data = np.insert(data, i, np.zeros_like(data[:, i, :]), axis=1)

    # Reshape to [num_examples, frequency bands (currrently 1), time, num_electrodes]
    data = np.einsum("bet->bte", data).reshape(data.shape[0], data.shape[2], 8, 8)
    data = np.expand_dims(data, axis=1)

    # Construct input dataset
    batch_size = preprocessor_params["foundation_model_batch_size"]
    foundation_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = torch.tensor(data[i : i + batch_size], dtype=torch.float32).to(
                device
            )
            batch_embeddings = model(
                batch, forward_features=True
            )  # Shape: [batch_size, 16]
            foundation_embeddings.append(batch_embeddings.cpu().numpy())

    foundation_embeddings = np.vstack(foundation_embeddings)

    return foundation_embeddings


@registry.register_config_setter("foundation_model_finetune_mlp")
def foundation_model_mlp_finetune_config_setter(
    experiment_config: ExperimentConfig, raws, _df_word
) -> ExperimentConfig:
    experiment_config.data_params.preprocessor_params = {}

    ch_names = sum([raw.info.ch_names for raw in raws], [])
    experiment_config.data_params.preprocessor_params["ch_names"] = ch_names

    # Setup foundation model config.
    if experiment_config.model_params["model_dir"]:
        ecog_config = create_video_mae_experiment_config_from_file(
            os.path.join(
                experiment_config.model_params["model_dir"], "experiment_config.ini"
            )
        )
        experiment_config.model_params["foundation_model_config"] = ecog_config
    else:
        ecog_config = dict_to_config(
            experiment_config.model_params["foundation_model_config"],
            VideoMAEExperimentConfig,
        )
        experiment_config.model_params["foundation_model_config"] = ecog_config

    # Set window width to whatever the sample length of the foundation model is.
    experiment_config.data_params.window_width = (
        ecog_config.ecog_data_config.sample_length
    )

    experiment_config.data_params.preprocessor_params["ecog_data_config"] = (
        ecog_config.ecog_data_config
    )

    return experiment_config


@registry.register_data_preprocessor("foundation_model_finetune_mlp")
def foundation_model_mlp_finetune_preprocessing_fn(data, preprocessor_params):
    data_config = preprocessor_params["ecog_data_config"]
    data = data.reshape(
        data.shape[0], data.shape[1], -1, data_config.original_fs // data_config.new_fs
    )
    data = data.mean(-1)

    for i in range(64):
        channel = "G" + str(i + 1)
        if not np.isin(channel, preprocessor_params["ch_names"]):
            data = np.insert(data, i, np.zeros_like(data[:, i, :]), axis=1)

    # Reshape to [num_examples, frequency bands (currrently 1), time, num_electrodes]
    data = np.einsum("bet->bte", data).reshape(data.shape[0], data.shape[2], 8, 8)
    data = np.expand_dims(data, axis=1)

    return data
