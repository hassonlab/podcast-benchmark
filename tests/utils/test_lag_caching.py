import torch
import torch.nn as nn
import pandas as pd

from core import registry
from core.config import BaseTaskConfig, ModelSpec, TaskConfig, TrainingParams
from utils.model_utils import build_model_from_spec
from utils.decoding_utils import train_decoding_model

import models.caching  # noqa: F401
import metrics  # noqa: F401


class CountingLatentEncoder(nn.Module):
    def __init__(self, counter):
        super().__init__()
        self.counter = counter

    def forward(self, x, **kwargs):
        self.counter["calls"] = self.counter.get("calls", 0) + x.shape[0]
        return x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)


class CountingDecoder(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.encoder_model = encoder_model
        self.head = nn.Linear(1, 1)

    def forward(self, x, cache_key=None, **kwargs):
        latents = self.encoder_model(x, cache_key=cache_key, **kwargs)
        return self.head(latents).squeeze(-1)


@registry.register_model_constructor("test_counting_encoder")
def _test_counting_encoder_constructor(model_params):
    return CountingLatentEncoder(counter=model_params["counter"])


@registry.register_model_constructor("test_counting_decoder")
def _test_counting_decoder_constructor(model_params):
    return CountingDecoder(encoder_model=model_params["encoder_model"])


def _counting_model_spec(counter):
    return ModelSpec(
        constructor_name="test_counting_decoder",
        sub_models={
            "encoder_model": ModelSpec(
                constructor_name="caching_model",
                sub_models={
                    "inner_model": ModelSpec(
                        constructor_name="test_counting_encoder",
                        params={"counter": counter},
                    )
                },
            )
        },
    )


class EmptyTaskConfig(BaseTaskConfig):
    input_fields: list[str] | None = None


def test_shared_cache_survives_fold_local_model_rebuilds():
    counter = {}
    spec = _counting_model_spec(counter)
    build_context = {"_cache_store": {}}

    model_one = build_model_from_spec(spec, lag=0, fold=1, build_context=build_context)
    model_two = build_model_from_spec(spec, lag=0, fold=2, build_context=build_context)

    x = torch.randn(3, 2, 4)
    keys = torch.tensor([0, 1, 2])

    model_one(x, cache_key=keys)
    model_two(x, cache_key=keys)

    assert counter["calls"] == 3
    assert len(build_context["_cache_store"]) == 3


def test_new_lag_uses_fresh_cache_store():
    counter = {}
    spec = _counting_model_spec(counter)

    first_context = {"_cache_store": {}}
    second_context = {"_cache_store": {}}

    x = torch.randn(2, 2, 4)
    keys = torch.tensor([0, 1])

    build_model_from_spec(spec, lag=0, fold=1, build_context=first_context)(x, cache_key=keys)
    build_model_from_spec(spec, lag=1000, fold=1, build_context=second_context)(x, cache_key=keys)

    assert counter["calls"] == 4
    assert len(first_context["_cache_store"]) == 2
    assert len(second_context["_cache_store"]) == 2


def test_train_decoding_model_reuses_cache_across_folds_within_lag(tmp_path):
    counter = {}
    spec = _counting_model_spec(counter)
    neural_data = torch.randn(6, 2, 4)
    targets = torch.randn(6)
    data_df = pd.DataFrame(index=range(6))

    task_config = TaskConfig(
        task_name="cache_test_task",
        task_specific_config=EmptyTaskConfig(input_fields=[]),
    )
    training_params = TrainingParams(
        batch_size=2,
        epochs=1,
        n_folds=2,
        fold_type="sequential_folds",
        losses=["mse"],
        metrics=[],
        early_stopping_metric="mse",
        smaller_is_better=True,
        tensorboard_logging=False,
    )

    _, _, cv_results = train_decoding_model(
        neural_data=neural_data,
        Y=targets,
        data_df=data_df,
        model_spec=spec,
        task_name="cache_test_task",
        task_config=task_config,
        lag=0,
        training_params=training_params,
        checkpoint_dir=str(tmp_path),
    )

    assert counter["calls"] == len(data_df)
    assert len(cv_results["fold_nums"]) == 2
