#!/usr/bin/env python3
"""Export the getting-started marimo notebook as a sequential Jupyter notebook."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tutorials" / "getting_started.py"
DEFAULT_OUTPUT = REPO_ROOT / "tutorials" / "getting_started.ipynb"


def _cell_with(cells, snippet):
    matches = [cell for cell in cells if snippet in cell.source]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exported cell containing {snippet!r}, found {len(matches)}"
        )
    return matches[0]


def _cell_starting_with(cells, prefix):
    matches = [cell for cell in cells if cell.source.lstrip().startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exported cell starting with {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def _source(value):
    return textwrap.dedent(value).strip() + "\n"


def export_notebook(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            str(SOURCE),
            "--sort",
            "top-down",
            "--no-include-outputs",
            "--output",
            str(output_path),
            "--force",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    notebook = nbformat.read(output_path, as_version=4)
    cells = notebook.cells

    setup_cell = nbformat.v4.new_code_cell(
        _source(
            """
            # Jupyter/Colab setup. When this notebook is opened outside a repository
            # checkout, clone the project and install its runtime dependencies.
            import importlib.util
            import os
            import subprocess
            import sys
            from pathlib import Path

            REPOSITORY_URL = "https://github.com/hassonlab/podcast-benchmark.git"
            IN_COLAB = "google.colab" in sys.modules
            start_dir = Path.cwd().resolve()
            repo_root = next(
                (
                    candidate
                    for candidate in (start_dir, *start_dir.parents)
                    if (candidate / "pyproject.toml").exists()
                ),
                start_dir,
            )

            if not (repo_root / "pyproject.toml").exists():
                clone_root = Path("/content") if IN_COLAB else repo_root
                candidate = clone_root / "podcast-benchmark"
                if not candidate.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", REPOSITORY_URL, str(candidate)],
                        check=True,
                    )
                repo_root = candidate.resolve()

            if IN_COLAB or importlib.util.find_spec("core") is None:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", str(repo_root)],
                    check=True,
                )

            os.chdir(repo_root)
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            print("Using the podcast-benchmark repository checkout.")
            """
        )
    )
    setup_cell.metadata["tags"] = ["setup"]
    cells.insert(0, setup_cell)

    imports_cell = _cell_with(cells, "import urllib.request")
    imports_cell.source = imports_cell.source.replace("import marimo as mo\n", "")
    imports_cell.source = imports_cell.source.replace(
        "repo_root = Path(__file__).resolve().parents[1]\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n\n",
        "",
    )
    imports_cell.source += "\n\n" + _source(
        """

        try:
            from IPython.display import display
        except ImportError:
            def display(value):
                print(value)
        """
    )

    _cell_with(cells, "demo_mode = mo.ui.checkbox").source = _source(
        """
        # Set this to False to reuse a complete Podcast dataset already on disk.
        USE_DEMO_DATA = True
        EXISTING_DATA_ROOT = repo_root / "data"

        print(
            "Demo mode will download subject 8 (~265 MB)."
            if USE_DEMO_DATA
            else f"Using existing data at {EXISTING_DATA_ROOT}."
        )
        """
    )

    _cell_with(cells, "not prepare_data.value").source = _source(
        """
        if USE_DEMO_DATA:
            data_root = repo_root / ".cache" / "tutorial_data"
            demo_file, downloaded = ensure_demo_data(data_root)
            action = "Downloaded" if downloaded else "Found"
            print(
                f"{action} {demo_file.relative_to(repo_root)} "
                f"({demo_expected_bytes / 1_000_000:.1f} MB)."
            )
        else:
            data_root = Path(EXISTING_DATA_ROOT).expanduser().resolve()
            expected_file = data_root / (
                "derivatives/ecogprep/sub-08/ieeg/"
                "sub-08_task-podcast_desc-highgamma_ieeg.fif"
            )
            if not expected_file.exists():
                raise FileNotFoundError(
                    "Subject 8 high-gamma data was not found under the selected root: "
                    f"{expected_file}"
                )
            print("Using the configured existing dataset.")
        """
    )

    for config_prefix in (
        "config = ExperimentConfig(",
        "synthetic_config = ExperimentConfig(",
    ):
        config_cell = _cell_starting_with(cells, config_prefix)
        config_cell.source = (
            config_cell.source.split("\nmo.md(", maxsplit=1)[0].rstrip() + "\n"
        )

    _cell_with(cells, 'start_benchmark = mo.ui.run_button').source = _source(
        """
        # This single call performs data loading, lag slicing, cross-validation,
        # training, checkpointing, and metric collection.
        benchmark_run = run_benchmark(config)
        result_path = Path(benchmark_run.output_dir).relative_to(repo_root)
        print(f"Finished. Results were written to {result_path}.")
        """
    )
    run_cell = _cell_with(cells, "not start_benchmark.value")
    cells.remove(run_cell)

    _cell_with(cells, "if column in benchmark_run.lag_results.columns").source = _source(
        """
        result_columns = [
            column
            for column in [
                "run_unit", "lags", "test_roc_auc_mean", "test_roc_auc_std"
            ]
            if column in benchmark_run.lag_results.columns
        ]
        display(benchmark_run.lag_results[result_columns])
        lag_figure, _lag_axes = plot_lag_results(
            benchmark_run.lag_results, metric="test_roc_auc"
        )
        display(lag_figure)
        """
    )

    _cell_with(cells, "prepare_synthetic = mo.ui.run_button").source = _source(
        """
        print("Generating the synthetic four-electrode Raw and label table...")
        """
    )

    synthetic_prepare = _cell_with(cells, "not prepare_synthetic.value")
    synthetic_prepare.source = _source(
        """
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

        print(
            f"Wrote {synthetic_raw_path.name} with {len(synthetic_raw.ch_names)} "
            f"electrodes and {int(synthetic_labels['sentence_onset'].sum())} "
            "sentence onsets."
        )
        display(synthetic_preview)
        display(synthetic_labels.head(12))
        """
    )

    _cell_with(cells, "run_synthetic_benchmark = mo.ui.run_button").source = _source(
        """
        synthetic_benchmark_run = run_benchmark(synthetic_config)
        synthetic_result_path = Path(
            synthetic_benchmark_run.output_dir
        ).relative_to(repo_root)
        print(f"Finished. Results were written to {synthetic_result_path}.")
        """
    )
    synthetic_run_cell = _cell_with(cells, "not run_synthetic_benchmark.value")
    cells.remove(synthetic_run_cell)

    _cell_with(
        cells, "if column in synthetic_benchmark_run.lag_results.columns"
    ).source = _source(
        """
        synthetic_result_columns = [
            column
            for column in [
                "run_unit", "lags", "test_roc_auc_mean", "test_roc_auc_std"
            ]
            if column in synthetic_benchmark_run.lag_results.columns
        ]
        display(synthetic_benchmark_run.lag_results[synthetic_result_columns])
        synthetic_lag_figure, _synthetic_lag_axes = plot_lag_results(
            synthetic_benchmark_run.lag_results, metric="test_roc_auc"
        )
        display(synthetic_lag_figure)
        """
    )

    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    notebook.metadata["podcast_benchmark"] = {
        "generated_from": "tutorials/getting_started.py",
        "generator": "scripts/export_tutorial_ipynb.py",
    }
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    nbformat.validate(notebook)
    nbformat.write(notebook, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_notebook(args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
