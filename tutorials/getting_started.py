import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from core import registry
    from core.config import (
        DataParams,
        ExperimentConfig,
        ModelSpec,
        TaskConfig,
        TrainingParams,
    )
    from main import run_benchmark
    from tasks.sentence_onset import SentenceOnsetConfig
    from utils.plot_utils import plot_lag_results

    return (
        DataParams,
        ExperimentConfig,
        ModelSpec,
        Path,
        SentenceOnsetConfig,
        TaskConfig,
        TrainingParams,
        mne,
        mo,
        nn,
        np,
        pd,
        plt,
        plot_lag_results,
        registry,
        repo_root,
        run_benchmark,
        torch,
        urllib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Your first NeuroCast Benchmark model

    This tutorial follows the same path as a full experiment—**neural data →
    preprocessor → registered model → cross-validated lag sweep**—with settings
    chosen to finish quickly on a CPU.

    You will decode whether a word begins a sentence from one participant's
    high-gamma ECoG activity. The only model-specific work is registering a
    preprocessor and a small PyTorch model. One `run_benchmark(config)` call then
    handles data loading, lagged windows, folds, training, metrics, checkpoints,
    and result files.

    To open this notebook locally:

    ```bash
    pip install -e ".[tutorial]"
    marimo edit tutorials/getting_started.py
    ```
    """)
    return


@app.cell
def _(mo, repo_root):
    demo_mode = mo.ui.checkbox(
        value=True,
        label="Use the one-subject demo (recommended)",
    )
    existing_data_root = mo.ui.text(
        value=str(repo_root / "data"),
        label="Existing full dataset root",
        full_width=True,
    )
    prepare_data = mo.ui.run_button(label="Prepare data")

    mo.vstack(
        [
            mo.md(
                """
                ## 1. Setup

                Demo mode downloads only subject 8's preprocessed high-gamma recording
                from OpenNeuro (about **265 MB**). Sentence labels are already included
                in this repository, so no podcast audio or language features are needed.

                Turn demo mode off to reuse a full Podcast dataset already on disk.
                """
            ),
            demo_mode,
            existing_data_root,
            prepare_data,
        ]
    )
    return demo_mode, existing_data_root, prepare_data


@app.cell
def _(Path, urllib):
    demo_relative_path = Path(
        "derivatives/ecogprep/sub-08/ieeg/"
        "sub-08_task-podcast_desc-highgamma_ieeg.fif"
    )
    demo_url = (
        "https://s3.amazonaws.com/openneuro.org/ds005574/"
        f"{demo_relative_path.as_posix()}"
    )
    demo_expected_bytes = 265_460_998

    def ensure_demo_data(destination_root):
        destination_root = Path(destination_root)
        destination = destination_root / demo_relative_path
        if destination.exists() and destination.stat().st_size == demo_expected_bytes:
            return destination, False

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(demo_url, partial)
            actual_bytes = partial.stat().st_size
            if actual_bytes != demo_expected_bytes:
                raise RuntimeError(
                    "Demo download was incomplete: "
                    f"expected {demo_expected_bytes:,} bytes, got {actual_bytes:,}."
                )
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return destination, True

    return demo_expected_bytes, ensure_demo_data


@app.cell
def _(
    Path,
    demo_expected_bytes,
    demo_mode,
    ensure_demo_data,
    existing_data_root,
    mo,
    prepare_data,
    repo_root,
):
    mo.stop(
        not prepare_data.value,
        mo.callout("Choose a data mode, then click **Prepare data**.", kind="info"),
    )

    if demo_mode.value:
        data_root = repo_root / ".cache" / "tutorial_data"
        with mo.status.spinner("Downloading or checking the subject 8 demo..."):
            demo_file, downloaded = ensure_demo_data(data_root)
        action = "Downloaded" if downloaded else "Found"
        setup_message = (
            f"{action} `{demo_file}` "
            f"({demo_expected_bytes / 1_000_000:.1f} MB)."
        )
    else:
        data_root = Path(existing_data_root.value).expanduser().resolve()
        expected_file = data_root / (
            "derivatives/ecogprep/sub-08/ieeg/"
            "sub-08_task-podcast_desc-highgamma_ieeg.fif"
        )
        if not expected_file.exists():
            raise FileNotFoundError(
                "Subject 8 high-gamma data was not found under the selected root: "
                f"{expected_file}"
            )
        setup_message = f"Using existing data at `{data_root}`."

    mo.callout(setup_message, kind="success")
    return (data_root,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Define and register the model

    Registry decorators are the benchmark's extension surface. The preprocessor
    below averages each neural window over time, producing one feature per
    electrode. The decoder is a single logistic-regression layer.

    In your own integration, replace these two functions while leaving the runner
    and task configuration unchanged.
    """)
    return


@app.cell
def _(nn, np, registry, torch):
    @registry.register_data_preprocessor("tutorial_temporal_mean")
    def tutorial_temporal_mean(data, _params):
        """Convert [events, electrodes, time] windows to electrode features."""
        return np.asarray(data, dtype=np.float32).mean(axis=-1)

    class TutorialLinearDecoder(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.linear = nn.Linear(input_channels, 1)

        def forward(self, features):
            probabilities = torch.sigmoid(self.linear(features))
            return probabilities.squeeze(-1)

    @registry.register_model_constructor("tutorial_linear_decoder")
    def build_tutorial_linear_decoder(model_params):
        return TutorialLinearDecoder(model_params["input_channels"])

    return


@app.cell
def _(
    DataParams,
    ExperimentConfig,
    ModelSpec,
    SentenceOnsetConfig,
    TaskConfig,
    TrainingParams,
    data_root,
    mo,
    repo_root,
):
    config = ExperimentConfig(
        model_spec=ModelSpec(
            constructor_name="tutorial_linear_decoder",
            params={"output_dim": 1},
        ),
        config_setter_name="set_input_channels",
        task_config=TaskConfig(
            task_name="sentence_onset_task",
            data_params=DataParams(
                data_root=str(data_root),
                subject_ids=[8],
                use_high_gamma=True,
                window_width=1.0,
                preprocessing_fn_name="tutorial_temporal_mean",
            ),
            task_specific_config=SentenceOnsetConfig(
                sentence_csv_path=str(
                    repo_root / "processed_data" / "all_sentences_podcast.csv"
                ),
                word_csv_path=str(
                    repo_root
                    / "processed_data"
                    / "df_word_onset_with_pos_class.csv"
                ),
                negatives_per_positive=1,
            ),
        ),
        training_params=TrainingParams(
            batch_size=512,
            epochs=20,
            learning_rate=0.05,
            weight_decay=0.001,
            early_stopping_patience=3,
            n_folds=2,
            # np.arange treats max_lag as exclusive, so 1750 includes +1500.
            min_lag=-1500,
            max_lag=1750,
            lag_step_size=250,
            losses=["bce"],
            loss_weights=[1.0],
            metrics=["roc_auc"],
            early_stopping_metric="roc_auc",
            smaller_is_better=False,
            tensorboard_logging=False,
            random_seed=42,
        ),
        trial_name="tutorial_sentence_onset",
        output_dir=str(repo_root / ".cache" / "tutorial_runs" / "results"),
        checkpoint_dir=str(
            repo_root / ".cache" / "tutorial_runs" / "checkpoints"
        ),
        tensorboard_dir=str(
            repo_root / ".cache" / "tutorial_runs" / "event_logs"
        ),
    )

    mo.md(
        f"""
        ## 3. Run the sentence-onset task

        This abbreviated configuration uses subject `{config.task_config.data_params.subject_ids[0]}`,
        `{config.training_params.n_folds}` folds, and lags from
        `{config.training_params.min_lag}` to `1500` ms in
        `{config.training_params.lag_step_size}` ms steps.

        The configuration is intentionally ordinary: the same `ExperimentConfig`
        fields can be supplied by YAML for larger reproducible runs.
        """
    )
    return (config,)


@app.cell
def _(mo):
    start_benchmark = mo.ui.run_button(label="Run benchmark")
    start_benchmark
    return (start_benchmark,)


@app.cell
def _(config, mo, run_benchmark, start_benchmark):
    mo.stop(
        not start_benchmark.value,
        mo.callout(
            "Click **Run benchmark** when you are ready. Training writes only under "
            "`.cache/tutorial_runs`.",
            kind="info",
        ),
    )
    with mo.status.spinner("Training the linear decoder across three lags..."):
        benchmark_run = run_benchmark(config)
    mo.callout(
        f"Finished. Results were written to `{benchmark_run.output_dir}`.",
        kind="success",
    )
    return (benchmark_run,)


@app.cell
def _(benchmark_run, mo, plot_lag_results):
    result_columns = [
        column
        for column in [
            "run_unit",
            "lags",
            "test_roc_auc_mean",
            "test_roc_auc_std",
        ]
        if column in benchmark_run.lag_results.columns
    ]
    results_table = mo.ui.table(
        benchmark_run.lag_results[result_columns],
        selection=None,
    )
    lag_figure, _lag_axes = plot_lag_results(
        benchmark_run.lag_results,
        metric="test_roc_auc",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Inspect performance across lags

                ROC-AUC is averaged over the two held-out folds. The shaded region is
                one fold standard deviation. With such a small tutorial run, focus on
                the workflow rather than treating the values as a benchmark estimate.
                """
            ),
            results_table,
            lag_figure,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2: Bring your own dataset

    The task, preprocessor, model, and runner are all independent of the Podcast
    dataset. To make that concrete, we will create a tiny synthetic recording with
    four ECoG electrodes and register a dataset getter for it.

    Sentence onsets occur at reproducibly irregular times. Neural activity follows
    a time-warped sinusoid whose peaks are locked to sentence onsets and fall away
    smoothly on either side. Noise and different electrode gains keep the example
    realistic enough to train while leaving the expected structure easy to see.
    """)
    return


@app.cell
def _(Path, mne, np, pd, registry):
    synthetic_dataset_name = "tutorial_synthetic_sentence_data"

    def make_synthetic_sentence_dataset(
        *, duration_s=180.0, sampling_rate=100.0, n_electrodes=4, seed=7
    ):
        """Create a Raw recording and canonical sentence-onset label table."""
        rng = np.random.default_rng(seed)
        n_samples = int(duration_s * sampling_rate)
        sample_times = np.arange(n_samples) / sampling_rate

        # Draw irregular word timings, then start sentences after randomly sized
        # groups of words. This makes the sentence periods non-isochronous.
        word_intervals = rng.uniform(0.55, 0.95, size=int(duration_s / 0.55))
        word_onsets = 2.0 + np.cumsum(word_intervals)
        word_onsets = word_onsets[word_onsets < duration_s - 2.0]
        sentence_positions = []
        position = int(rng.integers(4, 9))
        while position < len(word_onsets):
            sentence_positions.append(position)
            position += int(rng.integers(7, 14))

        sentence_mask = np.zeros(len(word_onsets), dtype=bool)
        sentence_mask[sentence_positions] = True
        sentence_onsets = word_onsets[sentence_mask]

        # Place a raised-cosine pulse at each onset. Its width adapts to the
        # neighboring sentence intervals, while remaining narrow enough to
        # produce a clear, symmetric falloff around the onset.
        onset_locked_wave = np.zeros(n_samples, dtype=np.float32)
        sentence_intervals = np.diff(sentence_onsets)
        for onset_index, onset in enumerate(sentence_onsets):
            previous_interval = sentence_intervals[max(onset_index - 1, 0)]
            next_interval = sentence_intervals[min(onset_index, len(sentence_intervals) - 1)]
            half_width = min(1.5, 0.35 * previous_interval, 0.35 * next_interval)
            in_pulse = np.abs(sample_times - onset) <= half_width
            phase = (sample_times[in_pulse] - onset) / half_width
            pulse = 0.5 * (1.0 + np.cos(np.pi * phase))
            onset_locked_wave[in_pulse] = np.maximum(
                onset_locked_wave[in_pulse], pulse.astype(np.float32)
            )

        electrode_gains = np.linspace(1.0, 0.35, n_electrodes, dtype=np.float32)
        noise_uv = rng.normal(0.0, 0.15, size=(n_electrodes, n_samples))
        slow_background = 0.08 * np.sin(2 * np.pi * 0.08 * sample_times)
        data_uv = (
            0.1
            + electrode_gains[:, None] * onset_locked_wave[None, :]
            + slow_background[None, :]
            + noise_uv
        ).astype(np.float32)

        channel_names = [f"SYNTH_ECOG_{index + 1}" for index in range(n_electrodes)]
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=sampling_rate,
            ch_types=["ecog"] * n_electrodes,
        )
        raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
        raw.info["description"] = (
            "Synthetic sentence data: onset-locked sinusoid plus Gaussian noise"
        )

        labels = pd.DataFrame(
            {
                "event_id": [
                    f"synthetic_word_{index:04d}"
                    for index in range(len(word_onsets))
                ],
                "start": word_onsets,
                "word": [f"word_{index}" for index in range(len(word_onsets))],
                "sentence_onset": sentence_mask,
            }
        )
        return raw, labels, sentence_onsets

    @registry.register_dataset(synthetic_dataset_name)
    def load_synthetic_sentence_raw(_subject_id, data_params):
        """Load the tutorial FIF and satisfy the benchmark dataset contract."""
        raw_path = Path(data_params.data_root) / data_params.dataset_params["raw_file"]
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Synthetic Raw file not found: {raw_path}. "
                "Click 'Generate synthetic dataset' first."
            )
        return mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    return make_synthetic_sentence_dataset, synthetic_dataset_name


@app.cell
def _(mo):
    prepare_synthetic = mo.ui.run_button(label="Generate synthetic dataset")
    prepare_synthetic
    return (prepare_synthetic,)


@app.cell
def _(make_synthetic_sentence_dataset, mo, plt, prepare_synthetic, repo_root):
    mo.stop(
        not prepare_synthetic.value,
        mo.callout(
            "Click **Generate synthetic dataset** to create the FIF and labels locally.",
            kind="info",
        ),
    )

    synthetic_data_root = repo_root / ".cache" / "tutorial_synthetic_data"
    synthetic_data_root.mkdir(parents=True, exist_ok=True)
    synthetic_raw_path = synthetic_data_root / "subject_1_sentence_raw.fif"
    synthetic_labels_path = synthetic_data_root / "sentence_labels.csv"

    synthetic_raw, synthetic_labels, synthetic_sentence_onsets = (
        make_synthetic_sentence_dataset()
    )
    synthetic_raw.save(synthetic_raw_path, overwrite=True, verbose=False)
    synthetic_labels.to_csv(synthetic_labels_path, index=False)

    preview_seconds = 45
    preview_samples = int(preview_seconds * synthetic_raw.info["sfreq"])
    synthetic_preview, synthetic_axes = plt.subplots(figsize=(9, 4))
    preview_times = synthetic_raw.times[:preview_samples]
    preview_data_uv = synthetic_raw.get_data(stop=preview_samples) * 1e6
    for electrode_index, values in enumerate(preview_data_uv):
        synthetic_axes.plot(
            preview_times,
            values + electrode_index * 1.8,
            linewidth=0.8,
            label=synthetic_raw.ch_names[electrode_index],
        )
    for sentence_onset in synthetic_sentence_onsets:
        if sentence_onset <= preview_seconds:
            synthetic_axes.axvline(
                sentence_onset, color="black", alpha=0.35, linewidth=1
            )
    synthetic_axes.set(
        xlabel="Time (s)",
        ylabel="Electrodes (offset for display)",
        title="Onset-locked neural waves; vertical lines are sentence onsets",
    )
    synthetic_axes.set_yticks([])
    synthetic_axes.grid(alpha=0.15)
    synthetic_preview.tight_layout()

    mo.vstack(
        [
            mo.callout(
                f"Wrote `{synthetic_raw_path.name}` with "
                f"{len(synthetic_raw.ch_names)} electrodes and "
                f"`{synthetic_labels_path.name}` with "
                f"{int(synthetic_labels['sentence_onset'].sum())} sentence onsets.",
                kind="success",
            ),
            synthetic_preview,
            mo.ui.table(synthetic_labels.head(12), selection=None),
        ]
    )
    return synthetic_data_root, synthetic_labels_path


@app.cell
def _(
    DataParams,
    ExperimentConfig,
    ModelSpec,
    SentenceOnsetConfig,
    TaskConfig,
    TrainingParams,
    mo,
    repo_root,
    synthetic_data_root,
    synthetic_dataset_name,
    synthetic_labels_path,
):
    synthetic_config = ExperimentConfig(
        # These are exactly the same registered model and preprocessor used above.
        model_spec=ModelSpec(
            constructor_name="tutorial_linear_decoder",
            params={"output_dim": 1},
        ),
        config_setter_name="set_input_channels",
        task_config=TaskConfig(
            # The task is unchanged; only its canonical label table is different.
            task_name="sentence_onset_task",
            data_params=DataParams(
                dataset_name=synthetic_dataset_name,
                dataset_params={"raw_file": "subject_1_sentence_raw.fif"},
                data_root=str(synthetic_data_root),
                subject_ids=[1],
                use_high_gamma=False,
                signal_unit="uV",
                window_width=1.0,
                preprocessing_fn_name="tutorial_temporal_mean",
            ),
            task_specific_config=SentenceOnsetConfig(
                labels_path=str(synthetic_labels_path),
                negatives_per_positive=1,
            ),
        ),
        training_params=TrainingParams(
            batch_size=128,
            epochs=20,
            learning_rate=0.05,
            weight_decay=0.001,
            early_stopping_patience=3,
            n_folds=2,
            # np.arange treats max_lag as exclusive, so 1750 includes +1500.
            min_lag=-1500,
            max_lag=1750,
            lag_step_size=250,
            losses=["bce"],
            loss_weights=[1.0],
            metrics=["roc_auc"],
            early_stopping_metric="roc_auc",
            smaller_is_better=False,
            tensorboard_logging=False,
            random_seed=42,
        ),
        trial_name="tutorial_synthetic_sentence_onset",
        output_dir=str(repo_root / ".cache" / "tutorial_runs" / "results"),
        checkpoint_dir=str(
            repo_root / ".cache" / "tutorial_runs" / "checkpoints"
        ),
        tensorboard_dir=str(
            repo_root / ".cache" / "tutorial_runs" / "event_logs"
        ),
    )

    mo.md(
        f"""
        ## Run the same benchmark on the new dataset

        The execution call is unchanged. In the configuration, we only point to:

        - registered dataset `{synthetic_config.task_config.data_params.dataset_name}`;
        - the generated FIF through `dataset_params`;
        - the generated canonical sentence-label CSV.

        The task remains `{synthetic_config.task_config.task_name}`, and the model
        constructor and preprocessor remain `tutorial_linear_decoder` and
        `tutorial_temporal_mean`.
        """
    )
    return (synthetic_config,)


@app.cell
def _(mo):
    run_synthetic_benchmark = mo.ui.run_button(
        label="Run benchmark on synthetic data"
    )
    run_synthetic_benchmark
    return (run_synthetic_benchmark,)


@app.cell
def _(mo, run_benchmark, run_synthetic_benchmark, synthetic_config):
    mo.stop(
        not run_synthetic_benchmark.value,
        mo.callout(
            "Click **Run benchmark on synthetic data** when ready.", kind="info"
        ),
    )
    with mo.status.spinner("Training the same decoder on the synthetic Raw..."):
        synthetic_benchmark_run = run_benchmark(synthetic_config)
    mo.callout(
        f"Finished. Results were written to `{synthetic_benchmark_run.output_dir}`.",
        kind="success",
    )
    return (synthetic_benchmark_run,)


@app.cell
def _(mo, plot_lag_results, synthetic_benchmark_run):
    synthetic_result_columns = [
        column
        for column in [
            "run_unit",
            "lags",
            "test_roc_auc_mean",
            "test_roc_auc_std",
        ]
        if column in synthetic_benchmark_run.lag_results.columns
    ]
    synthetic_results_table = mo.ui.table(
        synthetic_benchmark_run.lag_results[synthetic_result_columns],
        selection=None,
    )
    synthetic_lag_figure, _synthetic_lag_axes = plot_lag_results(
        synthetic_benchmark_run.lag_results,
        metric="test_roc_auc",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Synthetic lag results

                Because the synthetic waves peak at sentence onset and fall away on
                either side, performance should peak near lag 0 and decrease for
                earlier and later windows. Exact values vary in this small run.
                """
            ),
            synthetic_results_table,
            synthetic_lag_figure,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go next

    - Increase folds and use a finer lag step in [`docs/configuration.md`](../docs/configuration.md).
    - Adapt the registered model and preprocessor using
      [`docs/onboarding-model.md`](../docs/onboarding-model.md). There are more customization options to fit your use case.
    - Try another built-in target from [`docs/task-reference.md`](../docs/task-reference.md).
    - Add a dataset or task without changing the training loop using the registry
      APIs described in [`docs/api-reference.md`](../docs/api-reference.md).

    For a production run, move this `ExperimentConfig` into YAML and invoke
    `python main.py --config ...`; the notebook and CLI use the same pipeline.
    """)
    return


if __name__ == "__main__":
    app.run()
