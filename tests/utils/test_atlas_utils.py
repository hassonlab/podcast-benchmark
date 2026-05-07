import numpy as np
from types import ModuleType

from utils.atlas_utils import (
    _build_region_map_from_arrays,
    build_electrode_region_map,
    group_electrodes_by_region,
)


def test_group_electrodes_by_region_groups_by_exact_atlas_label():
    grouped = group_electrodes_by_region(
        ["A1", "A2", "B1"],
        ["L G_temp_sup-Lateral", "R G_temp_sup-Lateral", "Background"],
        {
            "temporal": ["L G_temp_sup-Lateral", "R G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        },
    )

    assert grouped == {"temporal": ["A1", "A2"]}


def test_build_region_map_from_arrays_constructs_region_subject_mapping():
    atlas_image = np.zeros((2, 1, 1), dtype=int)
    atlas_image[0, 0, 0] = 34
    atlas_image[1, 0, 0] = 15
    affine = np.eye(4)

    region_map = _build_region_map_from_arrays(
        {
            2: (["A1", "A2"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            5: (["B1"], np.array([[0.0, 0.0, 0.0]])),
        },
        atlas_image,
        affine,
        {
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        },
    )

    assert region_map == {
        "temporal": {2: ["A1"], 5: ["B1"]},
        "frontal": {2: ["A2"]},
    }


def test_build_electrode_region_map_uses_loaded_raw_channels(monkeypatch):
    class FakeImg:
        affine = np.eye(4)

        def get_fdata(self):
            atlas_image = np.zeros((2, 1, 1), dtype=int)
            atlas_image[0, 0, 0] = 34
            atlas_image[1, 0, 0] = 15
            return atlas_image

    class FakeDatasets:
        @staticmethod
        def fetch_atlas_destrieux_2009():
            return {"maps": "fake-atlas"}

    class FakeImage:
        @staticmethod
        def load_img(_path):
            return FakeImg()

    raws = [
        type(
            "Raw",
            (),
            {
                "ch_names": ["A1", "A2"],
                "info": {"chs": [{"loc": np.array([0.0, 0.0, 0.0])}, {"loc": np.array([0.001, 0.0, 0.0])}]},
            },
        )(),
        type(
            "Raw",
            (),
            {
                "ch_names": ["B1"],
                "info": {"chs": [{"loc": np.array([0.0, 0.0, 0.0])}]},
            },
        )(),
    ]

    fake_nilearn = ModuleType("nilearn")
    fake_nilearn.datasets = FakeDatasets
    fake_nilearn.image = FakeImage
    monkeypatch.setitem(__import__("sys").modules, "nilearn", fake_nilearn)

    region_map = build_electrode_region_map(
        subject_ids=[2, 5],
        raws=raws,
        region_groups={
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        },
    )

    assert region_map == {
        "temporal": {2: ["A1"], 5: ["B1"]},
        "frontal": {2: ["A2"]},
    }


def test_build_region_map_from_arrays_omits_empty_regions():
    atlas_image = np.zeros((1, 1, 1), dtype=int)
    atlas_image[0, 0, 0] = 34
    affine = np.eye(4)

    region_map = _build_region_map_from_arrays(
        {
            2: (["A1"], np.array([[0.0, 0.0, 0.0]])),
        },
        atlas_image,
        affine,
        {
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        },
    )

    assert region_map == {"temporal": {2: ["A1"]}}
