# Adding a Dataset

A dataset supplies continuous neural recordings as MNE Raw objects. Tasks remain independent: they supply canonical stimulus events and targets.

## Register a Raw getter

Place a module under `datasets/`; it is imported automatically.

```python
from core import registry

@registry.register_dataset("my_dataset")
def load_my_raw(subject_id, data_params):
    raw = ...  # mne.io.BaseRaw, with data available when accessed
    raw.info["bads"] = dataset_specific_bad_channels(subject_id)
    return raw
```

The shared loader subsequently applies `target_sr`, `do_drop_bads`, `signal_unit`, `per_subject_electrodes`, and `channel_reg_ex`.

Configure it with:

```yaml
task_config:
  data_params:
    dataset_name: my_dataset
    dataset_params:
      recording_variant: cleaned
    data_root: /path/to/data
    subject_ids: [1, 2]
```

## Subject-specific stimulus timing

If canonical stimulus time differs from each subject's neural clock, register an event mapper:

```python
def event_times(task_df, subject_id, data_params):
    # Return one neural-clock time in seconds for every task_df row.
    return map_stimulus_to_neural_time(task_df["start"], subject_id)

@registry.register_dataset("my_dataset", event_time_getter=event_times)
def load_my_raw(subject_id, data_params):
    ...
```

The mapper must preserve row count and order. The runtime intersects out-of-bounds and duplicate rows across included subjects before concatenating channels.

## Canonical task labels

The reusable word-task schema is a CSV or TSV with unique `event_id`, numeric `start` in stimulus seconds, and optional `end` and `word`. Add task columns such as `surprisal`, `surprisal_class`, `is_content`, `pos_class`, or `sentence_onset`, then configure the corresponding task with `labels_path`.

## Brain Treebank example

Create cleaned broadband and high-gamma artifacts for a shared movie:

```bash
python scripts/preprocess_brain_treebank.py \
  --data-root /path/to/brain-treebank \
  --movie Cars-2 \
  --subjects 1 2 3 \
  --representations broadband highgamma
```

Generate the canonical word-label table (after installing the spaCy model documented by the script):

```bash
python scripts/build_word_labels.py \
  --dataset brain_treebank \
  --transcript /path/to/brain-treebank/transcripts/Cars-2/features.csv \
  --out processed_data/brain_treebank/cars-2/word_labels.csv
```

See `configs/examples/brain_treebank/gpt_surprise.yml` for a standard-decoder run.
