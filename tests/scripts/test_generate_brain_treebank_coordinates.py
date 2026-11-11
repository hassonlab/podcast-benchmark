import json

import pandas as pd

from scripts.generate_brain_treebank_coordinates import generate_mni_sidecar


def test_generate_mni_sidecar_uses_raw_names_and_reports_missing(tmp_path):
    localization = tmp_path / "localization"
    localization.mkdir()
    pd.DataFrame(
        {
            "Subject": ["sub_3", "sub_3"],
            "Electrode": ["A1", "B2"],
            "X": [1.5, 4.5],
            "Y": [2.5, 5.5],
            "Z": [3.5, 6.5],
        }
    ).to_csv(localization / "elec_coords_full.csv", index=False)
    labels_dir = tmp_path / "electrode_labels" / "sub_3"
    labels_dir.mkdir(parents=True)
    (labels_dir / "electrode_labels.json").write_text(
        json.dumps(["A1", "B*2", "DC4"])
    )

    output_root = tmp_path / "generated"
    path, unmatched = generate_mni_sidecar(tmp_path, 3, output_root=output_root)
    result = pd.read_csv(path, sep="\t")

    assert result.to_dict("records") == [
        {"name": "A1", "x": 1.5, "y": 2.5, "z": 3.5},
        {"name": "B*2", "x": 4.5, "y": 5.5, "z": 6.5},
    ]
    assert unmatched == ["DC4"]
