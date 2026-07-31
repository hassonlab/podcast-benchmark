"""Generate per-subject MNI coordinate sidecars from Brain Treebank localization."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _normalized_channel(value: str) -> str:
    return re.sub(r"[#*]", "", str(value)).strip().upper()


def generate_mni_sidecar(
    data_root: Path,
    subject_id: int,
    overwrite: bool = False,
    output_root: Path | None = None,
):
    localization_path = data_root / "localization" / "elec_coords_full.csv"
    labels_path = (
        data_root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    )
    output_path = (
        (output_root or data_root)
        / f"sub-{subject_id:02d}"
        / "ieeg"
        / f"sub-{subject_id:02d}_space-MNI152NLin2009aSym_electrodes.tsv"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Coordinate sidecar exists: {output_path}")

    table = pd.read_csv(localization_path)
    required = {"Subject", "Electrode", "X", "Y", "Z"}
    missing = required - set(table)
    if missing:
        raise ValueError(f"Localization file is missing columns: {sorted(missing)}")
    rows = table.loc[table["Subject"].astype(str) == f"sub_{subject_id}"].copy()
    rows["normalized_name"] = rows["Electrode"].map(_normalized_channel)
    if rows["normalized_name"].duplicated().any():
        duplicates = rows.loc[rows["normalized_name"].duplicated(False), "Electrode"]
        raise ValueError(f"Duplicate localized electrodes: {duplicates.tolist()}")
    lookup = rows.set_index("normalized_name")

    with labels_path.open() as stream:
        labels = json.load(stream)
    if isinstance(labels, dict):
        labels = list(labels.values())

    output_rows = []
    unmatched = []
    for channel in map(str, labels):
        key = _normalized_channel(channel)
        if key not in lookup.index:
            unmatched.append(channel)
            continue
        row = lookup.loc[key]
        coords = pd.to_numeric(row[["X", "Y", "Z"]], errors="coerce")
        if coords.isna().any():
            unmatched.append(channel)
            continue
        output_rows.append(
            {"name": channel, "x": coords["X"], "y": coords["Y"], "z": coords["Z"]}
        )

    if not output_rows:
        raise ValueError(f"No localized electrodes matched subject {subject_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, sep="\t", index=False)
    return output_path, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/brain-treebank"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_data/brain_treebank/coordinates"),
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[3, 7, 10])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for subject_id in args.subjects:
        path, unmatched = generate_mni_sidecar(
            args.data_root,
            subject_id,
            overwrite=args.overwrite,
            output_root=args.output_root,
        )
        print(f"subject {subject_id}: {path} ({len(unmatched)} unlocalized channels)")
        if unmatched:
            print("  unlocalized: " + ", ".join(unmatched))


if __name__ == "__main__":
    main()
