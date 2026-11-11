"""Create cleaned Brain Treebank broadband/high-gamma MNE artifacts."""

import argparse
import sys
from pathlib import Path

# Support the documented ``python scripts/preprocess_brain_treebank.py`` entry point.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brain_treebank import (
    artifact_paths,
    attach_mni_coordinates,
    build_sync_table,
    load_source_raw,
    mark_corrupted_channels,
    resolve_movie_record,
)
from utils.raw_preprocessing import high_gamma_envelope, prepare_broadband_raw


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--movie", required=True)
    parser.add_argument("--subjects", nargs="+", required=True, type=int)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=("broadband", "highgamma"),
        default=("broadband", "highgamma"),
    )
    parser.add_argument("--target-sfreq", type=float, default=512.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def preprocess_subject(data_root, movie, subject_id, representations, target_sfreq, overwrite):
    if "highgamma" in representations and target_sfreq <= 400:
        raise ValueError(
            "High-gamma output requires --target-sfreq greater than 400 Hz "
            "for the 70-200 Hz band"
        )
    record = resolve_movie_record(data_root, subject_id, movie)
    source = load_source_raw(record, target_sfreq=target_sfreq)
    attach_mni_coordinates(source, data_root, subject_id)
    mark_corrupted_channels(source, data_root, subject_id)
    broadband = prepare_broadband_raw(source, target_sfreq=target_sfreq)
    sync = build_sync_table(record["timings_path"])

    representations = list(dict.fromkeys(representations))
    paths = {
        representation: artifact_paths(data_root, subject_id, movie, representation)
        for representation in representations
    }
    existing = [
        path
        for raw_path, sync_path in paths.values()
        for path in (raw_path, sync_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Artifact already exists for subject {subject_id}: {existing[0]}; "
            "pass --overwrite to replace it"
        )

    outputs = {}
    sync_written = False
    for representation in representations:
        raw_path, sync_path = paths[representation]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        output_raw = broadband if representation == "broadband" else high_gamma_envelope(broadband)
        output_raw.save(raw_path, overwrite=overwrite, verbose=False)
        if not sync_written:
            sync.to_csv(sync_path, sep="\t", index=False)
            sync_written = True
        outputs[representation] = raw_path
    return outputs


def main():
    args = build_parser().parse_args()
    for subject_id in args.subjects:
        outputs = preprocess_subject(
            args.data_root,
            args.movie,
            subject_id,
            args.representations,
            args.target_sfreq,
            args.overwrite,
        )
        for representation, path in outputs.items():
            print(f"subject {subject_id} {representation}: {Path(path)}")


if __name__ == "__main__":
    main()
