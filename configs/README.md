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
- within `baselines/`, the next directory is the model family
- within `foundation_models/`, the layout is `model/task/variant.yml`
- within `controls/`, the layout is `experiment_family/variant.yml`
- filenames inside a model directory should not repeat the parent directory name

Examples:

- `configs/baselines/neural_conv_decoder/gpt2.yml`
- `configs/controls/llm_decoding/no_brain_data.yml`
- `configs/examples/example_foundation_model/finetuning.yml`
- `configs/foundation_models/brainbert/word_embedding/supersubject.yml`
- `configs/foundation_models/legacy/popt_word_embedding.yml`
