"""Atlas-based electrode region utilities for the Destrieux 2009 parcellation."""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial import KDTree


# Placeholder for future atlas region definitions.
REGION_GROUPS: dict[str, list[str]] = {
    "EAC": [
        "L G_temp_sup-G_T_transv",
        "L G_temp_sup-Plan_polar",
        "L G_temp_sup-Lateral",
        "L S_temporal_sup",
        "L G_temp_sup-Plan_tempo",
        "L S_temporal_transverse",
    ],
    "MTG": [
        "L G_temporal_middle",
    ],
    "ITG": [
        "L G_oc-temp_lat-fusifor",
        "L S_oc-temp_lat",
        "L G_temporal_inf",
    ],
    "TP": [
        "L Pole_temporal",
    ],
    "IFG": [
        "L G_front_inf-Triangul",
        "L G_front_inf-Opercular",
        "L G_front_inf-Orbital",
        "L S_front_inf",
        "L Lat_Fis-ant-Vertical",
    ],
    "TPJ": [
        "L G_pariet_inf-Supramar",
        "L G_pariet_inf-Angular",
        "L Lat_Fis-post",
        "L S_intrapariet_and_P_trans",
    ],
    "PRC": [
        "L G_precentral",
        "L S_precentral-inf-part",
        "L S_precentral-sup-part",
    ],
    "PC": [
        "L G_postcentral",
        "L S_postcentral",
        "L G_and_S_subcentral",
        "L S_central",
        "L G_and_S_paracentral",
    ],
    "RIGHT": [
        "R G_postcentral",
        "R S_postcentral",
        "R S_central",
        "R G_and_S_paracentral",
        "R G_and_S_subcentral",
        "R G_precentral",
        "R G_pariet_inf-Angular",
        "R S_intrapariet_and_P_trans",
        "R Lat_Fis-post",
        "R S_precentral-inf-part",
        "R S_precentral-sup-part",
        "R G_pariet_inf-Supramar",
        "R G_front_inf-Orbital",
        "R G_front_inf-Opercular",
        "R G_front_inf-Triangul",
        "R G_oc-temp_lat-fusifor",
        "R G_temporal_inf",
        "R Lat_Fis-ant-Vertical",
        "R S_front_inf",
        "R S_oc-temp_lat",
        "R G_temporal_middle",
        "R S_temporal_sup",
        "R G_temp_sup-Lateral",
        "R G_temp_sup-G_T_transv",
        "R G_temp_sup-Plan_polar",
        "R G_temp_sup-Plan_tempo",
        "R Pole_temporal",
        "R S_temporal_transverse",
    ],
}


# Full 151-label Destrieux 2009 atlas label list (static, never changes).
# Index 0 = Background, indices 1-75 = L hemisphere, indices 76-150 = R hemisphere.
DESTRIEUX_2009_LABELS: list[str] = [
    "Background",
    "L G_and_S_frontomargin",
    "L G_and_S_occipital_inf",
    "L G_and_S_paracentral",
    "L G_and_S_subcentral",
    "L G_and_S_transv_frontopol",
    "L G_and_S_cingul-Ant",
    "L G_and_S_cingul-Mid-Ant",
    "L G_and_S_cingul-Mid-Post",
    "L G_cingul-Post-dorsal",
    "L G_cingul-Post-ventral",
    "L G_cuneus",
    "L G_front_inf-Opercular",
    "L G_front_inf-Orbital",
    "L G_front_inf-Triangul",
    "L G_front_middle",
    "L G_front_sup",
    "L G_Ins_lg_and_S_cent_ins",
    "L G_insular_short",
    "L G_occipital_middle",
    "L G_occipital_sup",
    "L G_oc-temp_lat-fusifor",
    "L G_oc-temp_med-Lingual",
    "L G_oc-temp_med-Parahip",
    "L G_orbital",
    "L G_pariet_inf-Angular",
    "L G_pariet_inf-Supramar",
    "L G_parietal_sup",
    "L G_postcentral",
    "L G_precentral",
    "L G_precuneus",
    "L G_rectus",
    "L G_subcallosal",
    "L G_temp_sup-G_T_transv",
    "L G_temp_sup-Lateral",
    "L G_temp_sup-Plan_polar",
    "L G_temp_sup-Plan_tempo",
    "L G_temporal_inf",
    "L G_temporal_middle",
    "L Lat_Fis-ant-Horizont",
    "L Lat_Fis-ant-Vertical",
    "L Lat_Fis-post",
    "L Medial_wall",
    "L Pole_occipital",
    "L Pole_temporal",
    "L S_calcarine",
    "L S_central",
    "L S_cingul-Marginalis",
    "L S_circular_insula_ant",
    "L S_circular_insula_inf",
    "L S_circular_insula_sup",
    "L S_collat_transv_ant",
    "L S_collat_transv_post",
    "L S_front_inf",
    "L S_front_middle",
    "L S_front_sup",
    "L S_interm_prim-Jensen",
    "L S_intrapariet_and_P_trans",
    "L S_oc_middle_and_Lunatus",
    "L S_oc_sup_and_transversal",
    "L S_occipital_ant",
    "L S_oc-temp_lat",
    "L S_oc-temp_med_and_Lingual",
    "L S_orbital_lateral",
    "L S_orbital_med-olfact",
    "L S_orbital-H_Shaped",
    "L S_parieto_occipital",
    "L S_pericallosal",
    "L S_postcentral",
    "L S_precentral-inf-part",
    "L S_precentral-sup-part",
    "L S_suborbital",
    "L S_subparietal",
    "L S_temporal_inf",
    "L S_temporal_sup",
    "L S_temporal_transverse",
    "R G_and_S_frontomargin",
    "R G_and_S_occipital_inf",
    "R G_and_S_paracentral",
    "R G_and_S_subcentral",
    "R G_and_S_transv_frontopol",
    "R G_and_S_cingul-Ant",
    "R G_and_S_cingul-Mid-Ant",
    "R G_and_S_cingul-Mid-Post",
    "R G_cingul-Post-dorsal",
    "R G_cingul-Post-ventral",
    "R G_cuneus",
    "R G_front_inf-Opercular",
    "R G_front_inf-Orbital",
    "R G_front_inf-Triangul",
    "R G_front_middle",
    "R G_front_sup",
    "R G_Ins_lg_and_S_cent_ins",
    "R G_insular_short",
    "R G_occipital_middle",
    "R G_occipital_sup",
    "R G_oc-temp_lat-fusifor",
    "R G_oc-temp_med-Lingual",
    "R G_oc-temp_med-Parahip",
    "R G_orbital",
    "R G_pariet_inf-Angular",
    "R G_pariet_inf-Supramar",
    "R G_parietal_sup",
    "R G_postcentral",
    "R G_precentral",
    "R G_precuneus",
    "R G_rectus",
    "R G_subcallosal",
    "R G_temp_sup-G_T_transv",
    "R G_temp_sup-Lateral",
    "R G_temp_sup-Plan_polar",
    "R G_temp_sup-Plan_tempo",
    "R G_temporal_inf",
    "R G_temporal_middle",
    "R Lat_Fis-ant-Horizont",
    "R Lat_Fis-ant-Vertical",
    "R Lat_Fis-post",
    "R Medial_wall",
    "R Pole_occipital",
    "R Pole_temporal",
    "R S_calcarine",
    "R S_central",
    "R S_cingul-Marginalis",
    "R S_circular_insula_ant",
    "R S_circular_insula_inf",
    "R S_circular_insula_sup",
    "R S_collat_transv_ant",
    "R S_collat_transv_post",
    "R S_front_inf",
    "R S_front_middle",
    "R S_front_sup",
    "R S_interm_prim-Jensen",
    "R S_intrapariet_and_P_trans",
    "R S_oc_middle_and_Lunatus",
    "R S_oc_sup_and_transversal",
    "R S_occipital_ant",
    "R S_oc-temp_lat",
    "R S_oc-temp_med_and_Lingual",
    "R S_orbital_lateral",
    "R S_orbital_med-olfact",
    "R S_orbital-H_Shaped",
    "R S_parieto_occipital",
    "R S_pericallosal",
    "R S_postcentral",
    "R S_precentral-inf-part",
    "R S_precentral-sup-part",
    "R S_suborbital",
    "R S_subparietal",
    "R S_temporal_inf",
    "R S_temporal_sup",
    "R S_temporal_transverse",
]


def slugify_region_name(region_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", region_name.lower()).strip("_")


def _lookup_atlas_labels(
    coords: np.ndarray,
    atlas_image: np.ndarray,
    affine: np.ndarray,
    atlas_labels: list[str],
) -> list[str]:
    image_label_coords = np.nonzero(atlas_image)
    i_arr, j_arr, k_arr = image_label_coords
    n_coords = len(i_arr)

    vox_homo = np.vstack(
        [
            i_arr.astype(float),
            j_arr.astype(float),
            k_arr.astype(float),
            np.ones(n_coords),
        ]
    )
    mni = (affine @ vox_homo)[:3].T

    tree = KDTree(mni)
    _, nearest_neighbor = tree.query(coords, k=1)

    x = image_label_coords[0][nearest_neighbor]
    y = image_label_coords[1][nearest_neighbor]
    z = image_label_coords[2][nearest_neighbor]
    elec_label_ids = atlas_image[x, y, z]
    return [atlas_labels[i] for i in elec_label_ids]


def group_electrodes_by_region(
    elec_names: list[str],
    atlas_labels: list[str],
    region_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    label_to_region: dict[str, str] = {}
    for region_name, labels in region_groups.items():
        for label in labels:
            label_to_region[label] = region_name

    result: dict[str, list[str]] = {}
    for elec_name, label in zip(elec_names, atlas_labels):
        region_name = label_to_region.get(label)
        if region_name is not None:
            result.setdefault(region_name, []).append(elec_name)

    return result


def _build_region_map_from_arrays(
    per_subject_data: dict[int, tuple[list[str], np.ndarray]],
    atlas_image: np.ndarray,
    affine: np.ndarray,
    region_groups: dict[str, list[str]],
) -> dict[str, dict[int, list[str]]]:
    result: dict[str, dict[int, list[str]]] = {}

    for subject_id, (elec_names, coords) in per_subject_data.items():
        labels = _lookup_atlas_labels(
            coords, atlas_image, affine, DESTRIEUX_2009_LABELS
        )
        grouped = group_electrodes_by_region(elec_names, labels, region_groups)
        for region_name, electrodes in grouped.items():
            result.setdefault(region_name, {})[subject_id] = electrodes

    return result


def build_electrode_region_map(
    subject_ids: list[int],
    raws: list[object],
    region_groups: dict[str, list[str]],
    atlas_path: str | None = None,
) -> dict[str, dict[int, list[str]]]:
    from nilearn import datasets, image as nli_image

    if atlas_path is None:
        atlas = datasets.fetch_atlas_destrieux_2009()
        atlas_path = atlas["maps"]

    img = nli_image.load_img(atlas_path)
    atlas_image = img.get_fdata().astype(int)
    affine = img.affine

    per_subject_data: dict[int, tuple[list[str], np.ndarray]] = {}
    for subject_id, raw in zip(subject_ids, raws):
        coords = np.array([ch["loc"][:3] * 1000 for ch in raw.info["chs"]])
        per_subject_data[subject_id] = (list(raw.ch_names), coords)

    return _build_region_map_from_arrays(
        per_subject_data,
        atlas_image,
        affine,
        region_groups,
    )
