# Brain Treebank model configs

The model-family directories each contain the same five per-subject tasks supported
by the canonical Brain Treebank word-label table for the Cars 2 recordings from
subjects 3, 7, and 10:

- `neural_conv_decoder/` uses 512 Hz high-gamma input.
- `linear_model/` applies a regularized linear readout to averaged 512 Hz high gamma.
- `brainbert/` uses 2048 Hz broadband input followed by STFT preprocessing.
- `diver/` uses broadband input resampled to 500 Hz and scaled to microvolts.
- `popt/` uses 2048 Hz broadband input, STFT preprocessing, and MNI coordinates.

All configs expect FIF/sync artifacts under `data/brain-treebank` and labels at
`processed_data/brain_treebank/cars-2/word_labels.csv`. Foundation model checkpoint
paths match the shared GPFS locations used by the corresponding Podcast configs.

Run any task with the normal entry point, for example:

```bash
python main.py --config configs/brain_treebank/brainbert/pos.yml
python main.py --config configs/brain_treebank/diver/gpt_surprise.yml
python main.py --config configs/brain_treebank/popt/sentence_onset.yml
```

Each config evaluates 21 lags from -1000 ms through +1000 ms in 100 ms steps.
`max_lag` is set to 1100 because the training runner treats that bound as exclusive.

Foundation configs are generated from the established Podcast subject configs by
`scripts/generate_brain_treebank_foundation_configs.py`.
