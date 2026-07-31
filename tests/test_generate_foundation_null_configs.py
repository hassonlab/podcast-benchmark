from scripts.generate_foundation_random_init_configs import build_random_init_config
from scripts.generate_foundation_shuffled_target_configs import (
    build_shuffled_target_config,
)


def _template():
    return {
        "trial_name": "brainbert_task",
        "task_config": {
            "data_params": {
                "preprocessor_params": [
                    {},
                    {
                        "foundation_model_spec": {
                            "constructor_name": "brainbert_finetune",
                            "random_init": False,
                        }
                    },
                ]
            }
        },
        "training_params": {"shuffle_targets": False},
    }


def test_random_init_config_uses_ten_repetitions_at_default_single_lag():
    config = build_random_init_config(_template())
    foundation_spec = config["task_config"]["data_params"]["preprocessor_params"][1][
        "foundation_model_spec"
    ]
    assert foundation_spec["random_init"] is True
    assert config["training_params"]["num_null_repetitions"] == 10
    assert config["training_params"]["lag"] == 0
    assert config["training_params"]["shuffle_targets"] is False


def test_shuffled_target_config_uses_one_hundred_repetitions_without_random_init():
    config = build_shuffled_target_config(_template())
    foundation_spec = config["task_config"]["data_params"]["preprocessor_params"][1][
        "foundation_model_spec"
    ]
    assert foundation_spec["random_init"] is False
    assert config["training_params"]["shuffle_targets"] is True
    assert config["training_params"]["num_null_repetitions"] == 100
    assert config["training_params"]["lag"] == 0
