import ast
import subprocess
import sys
from pathlib import Path

import mne
import nbformat
import numpy as np
import pandas as pd

from core import registry
from core.config import DataParams, TaskConfig
from tasks.sentence_onset import SentenceOnsetConfig, sentence_onset_task
from utils.data_utils import load_raws


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "tutorials" / "getting_started.py"
JUPYTER_NOTEBOOK = REPO_ROOT / "tutorials" / "getting_started.ipynb"


def _execute_cell_containing(definition_name, namespace):
    tree = ast.parse(NOTEBOOK.read_text())
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        definitions = {
            node.name
            for node in cell.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        if definition_name not in definitions:
            continue
        body = [node for node in cell.body if not isinstance(node, ast.Return)]
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(module, str(NOTEBOOK), "exec"), namespace)
        return namespace
    raise AssertionError(f"Notebook definition not found: {definition_name}")


def test_getting_started_notebook_is_valid_python():
    compile(NOTEBOOK.read_text(), str(NOTEBOOK), "exec")


def test_exported_jupyter_notebook_is_sequential_and_executed():
    notebook = nbformat.read(JUPYTER_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    serialized_notebook = JUPYTER_NOTEBOOK.read_text()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert notebook.metadata["podcast_benchmark"]["generated_from"] == (
        "tutorials/getting_started.py"
    )
    assert "google.colab" in code
    assert "benchmark_run = run_benchmark(config)" in code
    assert "synthetic_benchmark_run = run_benchmark(synthetic_config)" in code
    assert "mo.ui" not in code
    assert "mo.stop" not in code
    assert "__file__" not in code
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert outputs
    assert not [output for output in outputs if output.output_type == "error"]
    for unwanted_text in (
        "handrail6120",
        "/home/",
        "IProgress not found",
        "neural_tensor shape",
        "Train indices:",
        "Validation indices:",
        "Test indices:",
    ):
        assert unwanted_text not in serialized_notebook


def test_jupyter_export_is_reproducible(tmp_path):
    generated = tmp_path / "getting_started.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_tutorial_ipynb.py"),
            "--output",
            str(generated),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = nbformat.read(JUPYTER_NOTEBOOK, as_version=4)
    regenerated = nbformat.read(generated, as_version=4)
    assert [cell.cell_type for cell in regenerated.cells] == [
        cell.cell_type for cell in tracked.cells
    ]
    assert [cell.source for cell in regenerated.cells] == [
        cell.source for cell in tracked.cells
    ]


def test_getting_started_notebook_gates_expensive_actions():
    source = NOTEBOOK.read_text()

    assert 'prepare_data = mo.ui.run_button(label="Prepare data")' in source
    assert 'start_benchmark = mo.ui.run_button(label="Run benchmark")' in source
    assert (
        'prepare_synthetic = mo.ui.run_button(label="Generate synthetic dataset")'
        in source
    )
    assert 'label="Run benchmark on synthetic data"' in source
    assert "not prepare_data.value" in source
    assert "not start_benchmark.value" in source
    assert "not prepare_synthetic.value" in source
    assert "not run_synthetic_benchmark.value" in source


def test_synthetic_tutorial_dataset_satisfies_raw_and_task_contracts(tmp_path):
    namespace = {
        "Path": Path,
        "mne": mne,
        "np": np,
        "pd": pd,
        "registry": registry,
    }
    _execute_cell_containing("make_synthetic_sentence_dataset", namespace)

    raw, labels, sentence_onsets = namespace[
        "make_synthetic_sentence_dataset"
    ]()

    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 4
    assert raw.info["sfreq"] == 100.0
    assert labels["event_id"].is_unique
    assert labels["sentence_onset"].sum() == len(sentence_onsets)
    assert np.std(np.diff(sentence_onsets)) > 1.0

    data_uv = raw.get_data() * 1e6
    onset_samples = np.rint(sentence_onsets[1:] * raw.info["sfreq"]).astype(int)
    shoulder_offset = int(raw.info["sfreq"])
    at_peak = data_uv[:, onset_samples].mean()
    before_peak = data_uv[:, onset_samples - shoulder_offset].mean()
    after_peak = data_uv[:, onset_samples + shoulder_offset].mean()
    assert at_peak > max(before_peak, after_peak) + 0.25
    assert abs(before_peak - after_peak) < 0.15

    raw_path = tmp_path / "subject_1_sentence_raw.fif"
    labels_path = tmp_path / "sentence_labels.csv"
    raw.save(raw_path, overwrite=True, verbose=False)
    labels.to_csv(labels_path, index=False)

    data_params = DataParams(
        dataset_name=namespace["synthetic_dataset_name"],
        dataset_params={"raw_file": raw_path.name},
        data_root=str(tmp_path),
        subject_ids=[1],
    )
    loaded_raws = load_raws(data_params)
    assert len(loaded_raws) == 1
    assert loaded_raws[0].ch_names == raw.ch_names

    task_df = sentence_onset_task(
        TaskConfig(
            task_name="sentence_onset_task",
            data_params=data_params,
            task_specific_config=SentenceOnsetConfig(
                labels_path=str(labels_path), negatives_per_positive=1
            ),
        )
    )
    assert set(task_df["target"]) == {0.0, 1.0}
    assert int((task_df["target"] == 0).sum()) == int(
        (task_df["target"] == 1).sum()
    )
