# Config Layout

The `configs/` tree follows a category-first naming rule.

Top-level directories:

- `baselines/`: benchmark models trained directly on podcast-benchmark data
- `controls/`: control and ablation experiments
- `examples/`: onboarding and reference example configs
- `foundation_models/`: integrated upstream foundation-model experiments
- `hpo/`: hyperparameter search grids and tuning utilities

Naming rules:

- top-level directories are categories, not model names mixed with utilities
- within `baselines/`, the layout is `task/baseline_family/variant.yml`
- within `foundation_models/`, the layout is `model/task/variant.yml`
- within `controls/`, the layout is `experiment_family/variant.yml`

Examples:

- `configs/baselines/word_embedding_decoding_task/neural_conv_decoder/gpt2_supersubject.yml`
- `configs/baselines/word_embedding_decoding_task/ridge_regression/per_region.yml`
- `configs/baselines/volume_level_decoding_task/time_pooling_model_torch_ridge_per_region.yml`
- `configs/controls/llm_decoding/no_brain_data.yml`
- `configs/examples/example_foundation_model/finetuning.yml`
- `configs/foundation_models/brainbert/word_embedding/supersubject.yml`
- `configs/foundation_models/legacy/popt_word_embedding.yml`
