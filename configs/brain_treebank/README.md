# Brain Treebank model configs

The model-family directories each contain the same five per-subject tasks supported
by the canonical Brain Treebank word-label table for the Cars 2 recordings from
subjects 3, 7, and 10. Every YAML contains exactly one subject so jobs can run in
parallel; the subject-3 file keeps the short task name and subjects 7 and 10 use
`_subject7` and `_subject10` suffixes:

- `neural_conv_decoder/` uses 512 Hz high-gamma input.
- `linear_model/` applies a regularized linear readout to averaged 512 Hz high gamma.
- `brainbert/` uses 2048 Hz broadband input followed by STFT preprocessing.
- `diver/` uses broadband input resampled to 500 Hz and scaled to microvolts.
- `popt/` uses 2048 Hz broadband input and STFT preprocessing. Its LIP positional
  encoding is disabled because the public localization table has MNI XYZ but not
  the integer Left/Inferior/Posterior indices used by the pretrained checkpoint.

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

Preview and then submit all 45 single-subject foundation jobs with:

```bash
make train-brain-treebank-foundations DRY_RUN=1
make train-brain-treebank-foundations
```

Use `BTB_FOUNDATION_MODELS='brainbert diver'` to select model families and
`SBATCH_FLAGS='-p gpu --time=20:00:00'` for site-specific Slurm overrides.

Foundation configs are generated from the established Podcast subject configs by
`scripts/generate_brain_treebank_foundation_configs.py`.

Generate the MNI sidecars required by DIVER from the public localization table with:

```bash
python scripts/generate_brain_treebank_coordinates.py --subjects 3 7 10
```

The generated files live under `processed_data/brain_treebank/coordinates`, separately
from a potentially read-only raw-data mount. Unlocalized contacts are reported and
omitted; DIVER masks those missing coordinates.
