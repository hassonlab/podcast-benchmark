from core.config import DataParams, ExperimentConfig, ModelSpec, TaskConfig, TrainingParams
from models.brainbert.integration import set_finetuning_config as set_brainbert_finetuning_config
from models.diver.integration import set_diver_finetuning_config
from models.popt.integration import set_finetuning_config as set_popt_finetuning_config


class MockRaw:
    def __init__(self, ch_names, sfreq):
        self.ch_names = ch_names
        self.info = {"sfreq": sfreq}


def _cached_encoder_spec(top_level_constructor_name, encoder_constructor_name, params):
    return ModelSpec(
        constructor_name=top_level_constructor_name,
        params=params,
        sub_models={
            "encoder_model": ModelSpec(
                constructor_name="caching_model",
                sub_models={
                    "inner_model": ModelSpec(
                        constructor_name=encoder_constructor_name,
                    )
                },
            )
        },
    )


def test_brainbert_cached_encoder_config_is_populated():
    config = ExperimentConfig(
        model_spec=_cached_encoder_spec(
            "brainbert_finetune",
            "brainbert_encoder",
            {
                "model_dir": "models/brainbert/pretrained_model",
                "freeze_foundation": True,
                "dropout": 0.0,
                "mlp_layer_sizes": [],
            },
        ),
        task_config=TaskConfig(
            task_name="word_embedding_decoding_task",
            data_params=DataParams(window_width=1.0, target_sr=2048),
        ),
        training_params=TrainingParams(losses=["mse"]),
    )

    set_brainbert_finetuning_config(config, [MockRaw(["A", "B", "C"], 2048)], None)

    inner_spec = config.model_spec.sub_models["encoder_model"].sub_models["inner_model"]
    assert inner_spec.params["model_dir"] == "models/brainbert/pretrained_model"
    assert inner_spec.params["freeze_foundation"] is True
    assert inner_spec.params["input_channels"] == 40
    assert inner_spec.params["output_dim"] == 50


def test_popt_cached_encoder_config_is_populated():
    config = ExperimentConfig(
        model_spec=_cached_encoder_spec(
            "popt_finetune",
            "popt_encoder",
            {
                "model_dir": "models/popt/pretrained_model",
                "freeze_foundation": True,
                "dropout": 0.0,
                "mlp_layer_sizes": [],
                "use_brainbert": True,
                "use_lip_coords": True,
            },
        ),
        task_config=TaskConfig(
            task_name="word_embedding_decoding_task",
            data_params=DataParams(window_width=1.0, target_sr=2048, use_lip_coords=True),
        ),
        training_params=TrainingParams(losses=["mse"], batch_size=4),
    )

    set_popt_finetuning_config(config, [MockRaw(["A", "B"], 2048)], None)

    inner_spec = config.model_spec.sub_models["encoder_model"].sub_models["inner_model"]
    assert inner_spec.params["model_dir"] == "models/popt/pretrained_model"
    assert inner_spec.params["freeze_foundation"] is True
    assert inner_spec.params["input_channels"] == 40
    assert inner_spec.params["use_lip_coords"] is True
    assert inner_spec.params["output_dim"] == 50


def test_diver_cached_encoder_config_is_populated():
    config = ExperimentConfig(
        model_spec=_cached_encoder_spec(
            "diver_finetune",
            "diver_encoder",
            {
                "foundation_dir": "models/diver/pretrained_model/256_mp_rank_00_model_states.pt",
                "freeze_foundation": True,
                "patch_sampling_rate": 500,
                "patch_size": 50,
                "d_model": 256,
                "e_layer": 12,
                "ft_config": "flatten_linear",
            },
        ),
        task_config=TaskConfig(
            task_name="word_embedding_decoding_task",
            data_params=DataParams(window_width=1.0, target_sr=500),
        ),
        training_params=TrainingParams(losses=["mse"]),
    )

    set_diver_finetuning_config(config, [MockRaw(["A", "B", "C"], 500)], None)

    inner_spec = config.model_spec.sub_models["encoder_model"].sub_models["inner_model"]
    assert inner_spec.params["foundation_dir"].endswith("256_mp_rank_00_model_states.pt")
    assert inner_spec.params["freeze_foundation"] is True
    assert inner_spec.params["input_channels"] == 3
    assert inner_spec.params["output_dim"] == 50
