"""Plot electrodes assigned to atlas region groups.

This script uses the lightweight BIDS electrode localization sidecars in
``data/sub-*/ieeg``. It does not download or read the large neural data files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.atlas_utils import (  # noqa: E402
    DESTRIEUX_2009_LABELS,
    REGION_GROUPS,
    _lookup_atlas_labels,
    slugify_region_name,
)

OPENNEURO_BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds005574"
REGION_LEVEL_ORDER = (
    "EAC",
    "PC",
    "PRC",
    "IFG",
    "MTG",
    "ITG",
    "TPJ",
    "TP",
    "RIGHT",
)
LOCALIZATION_PATTERNS = (
    r"sub-\d+/ieeg/sub-\d+_space-MNI152NLin2009aSym_electrodes\.tsv$",
    r"sub-\d+/ieeg/sub-\d+_space-MNI152NLin2009aSym_coordsystem\.json$",
    r"sub-\d+/ieeg/sub-\d+_task-podcast_channels\.tsv$",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize electrodes assigned to utils.atlas_utils.REGION_GROUPS."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "processed_data" / "atlas_region_visualization",
    )
    parser.add_argument(
        "--nilearn-data-dir",
        type=Path,
        default=None,
        help="Directory for Nilearn atlas cache. Defaults to <output-dir>/nilearn_data.",
    )
    parser.add_argument(
        "--region-groups-json",
        type=Path,
        help="Optional JSON mapping of region name to Destrieux labels. Defaults to REGION_GROUPS.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload localization sidecars even when local files already exist.",
    )
    parser.add_argument(
        "--include-bad",
        action="store_true",
        help="Include channels marked bad in *_channels.tsv. Default keeps only good channels.",
    )
    parser.add_argument(
        "--allow-empty-region-groups",
        action="store_true",
        help="Accepted for compatibility; empty region groups are plotted as unassigned.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_region_groups(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return REGION_GROUPS

    with path.open() as f:
        region_groups = json.load(f)

    if not isinstance(region_groups, dict):
        raise ValueError("--region-groups-json must contain an object mapping names to labels")

    return {str(name): list(labels) for name, labels in region_groups.items()}


def discover_data_paths(data_root: Path) -> list[Path]:
    git_dir = data_root / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "-C", str(data_root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [Path(line) for line in result.stdout.splitlines() if line]
    else:
        paths = [path.relative_to(data_root) for path in data_root.rglob("*") if path.is_file()]

    regexes = [re.compile(pattern) for pattern in LOCALIZATION_PATTERNS]
    return sorted(path for path in paths if any(regex.search(path.as_posix()) for regex in regexes))


def download_missing_files(data_root: Path, rel_paths: list[Path], force: bool) -> None:
    for rel_path in rel_paths:
        local_path = data_root / rel_path
        if local_path.exists() and local_path.stat().st_size > 0 and not force:
            continue

        if local_path.is_symlink():
            local_path.unlink()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{OPENNEURO_BASE_URL}/{rel_path.as_posix()}"
        print(f"Downloading {url} -> {local_path}")
        urllib.request.urlretrieve(url, local_path)


def read_good_channel_names(channels_path: Path) -> set[str] | None:
    if not channels_path.exists():
        return None

    channels = pd.read_csv(channels_path, sep="\t")
    if "status" not in channels or "name" not in channels:
        return None

    return set(channels.loc[channels["status"].fillna("").str.lower() == "good", "name"])


def load_electrodes(data_root: Path, include_bad: bool) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for electrodes_path in sorted(
        data_root.glob("sub-*/ieeg/*_space-MNI152NLin2009aSym_electrodes.tsv")
    ):
        subject = electrodes_path.parts[-3]
        elecs = pd.read_csv(electrodes_path, sep="\t")
        required = {"name", "x", "y", "z"}
        missing = required - set(elecs.columns)
        if missing:
            raise ValueError(f"{electrodes_path} is missing columns: {sorted(missing)}")

        elecs = elecs.copy()
        elecs["subject"] = subject
        for axis in ("x", "y", "z"):
            elecs[axis] = pd.to_numeric(elecs[axis], errors="coerce")
        elecs = elecs.dropna(subset=["x", "y", "z"])

        if not include_bad:
            channels_path = electrodes_path.with_name(f"{subject}_task-podcast_channels.tsv")
            good_names = read_good_channel_names(channels_path)
            if good_names is not None:
                elecs = elecs[elecs["name"].isin(good_names)]

        rows.append(elecs[["subject", "name", "x", "y", "z"]])

    if not rows:
        raise FileNotFoundError(f"No MNI electrode TSV files found under {data_root}")

    return pd.concat(rows, ignore_index=True)


def load_atlas(
    atlas_path: str | None = None,
    nilearn_data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from nilearn import datasets, image as nli_image

    if atlas_path is None:
        fetch_kwargs = {}
        if nilearn_data_dir is not None:
            nilearn_data_dir.mkdir(parents=True, exist_ok=True)
            fetch_kwargs["data_dir"] = str(nilearn_data_dir)
        atlas_path = datasets.fetch_atlas_destrieux_2009(**fetch_kwargs)["maps"]

    img = nli_image.load_img(atlas_path)
    return img.get_fdata().astype(int), img.affine


def assign_region_groups(
    electrodes: pd.DataFrame,
    region_groups: dict[str, list[str]],
    nilearn_data_dir: Path | None = None,
) -> pd.DataFrame:
    electrodes = electrodes.copy()
    if not region_groups:
        electrodes["atlas_label"] = "unassigned"
        electrodes["region_group"] = "unassigned"
        return electrodes

    atlas_image, affine = load_atlas(nilearn_data_dir=nilearn_data_dir)
    coords = electrodes[["x", "y", "z"]].to_numpy(float)
    electrodes["atlas_label"] = _lookup_atlas_labels(
        coords, atlas_image, affine, DESTRIEUX_2009_LABELS
    )

    label_to_group = {
        label: region_name
        for region_name, labels in region_groups.items()
        for label in labels
    }
    electrodes["region_group"] = electrodes["atlas_label"].map(label_to_group).fillna("unassigned")
    return electrodes


def write_summaries(electrodes: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    assignments_path = output_dir / "electrode_region_assignments.csv"
    counts_path = output_dir / "region_group_counts.csv"

    electrodes.to_csv(assignments_path, index=False)

    counts = (
        electrodes.groupby(["region_group", "subject"], dropna=False)
        .size()
        .rename("n_electrodes")
        .reset_index()
        .sort_values(["region_group", "subject"])
    )
    counts.to_csv(counts_path, index=False)
    return assignments_path, counts_path


def region_sort_key(region: str) -> tuple[int, str]:
    if region in REGION_LEVEL_ORDER:
        return (REGION_LEVEL_ORDER.index(region), region)
    return (len(REGION_LEVEL_ORDER), region)


def display_region_name(region: str) -> str:
    return "STG" if region == "EAC" else region


def create_electrode_region_figure(electrodes: pd.DataFrame) -> plt.Figure:
    from nilearn import plotting

    electrodes = electrodes[electrodes["region_group"] != "unassigned"].copy()
    counts = (
        electrodes["region_group"]
        .value_counts()
        .sort_values(ascending=False, kind="mergesort")
    )
    groups = list(counts.index)
    cmap = plt.get_cmap("tab20", max(len(groups), 1))
    colors = {group: to_hex(cmap(i)) for i, group in enumerate(groups)}

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.15, 0.95],
        left=0.04,
        right=0.985,
        bottom=0.18,
        top=0.86,
        wspace=0.18,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]
    bar_ax = fig.add_subplot(grid[0, 2])

    views = [
        ("Left", "l"),
        ("Right", "r"),
    ]

    marker_coords = electrodes[["x", "y", "z"]].to_numpy(float)
    for ax, (title, display_mode) in zip(axes, views):
        display = plotting.plot_glass_brain(
            None,
            display_mode=display_mode,
            colorbar=False,
            figure=fig,
            axes=ax,
            title=title,
            black_bg=False,
            annotate=True,
        )
        for group in groups:
            mask = electrodes["region_group"] == group
            display.add_markers(
                marker_coords[mask],
                marker_color=colors[group],
                marker_size=34,
                alpha=0.9,
            )

    if counts.empty:
        bar_ax.text(
            0.5,
            0.5,
            "No electrodes matched REGION_GROUPS",
            ha="center",
            va="center",
            transform=bar_ax.transAxes,
            fontsize=13,
        )
        bar_ax.set_axis_off()
    else:
        bar_colors = [colors[group] for group in counts.index]
        x = np.arange(len(counts))
        bar_ax.bar(x, counts.values, color=bar_colors)
        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(
            [display_region_name(str(group)) for group in counts.index],
            rotation=45,
            ha="right",
        )
        bar_ax.set_ylabel("Electrodes")
        bar_ax.set_title("Electrodes per region")
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        for idx, value in enumerate(counts.values):
            bar_ax.text(
                idx,
                value + max(counts.max() * 0.02, 0.2),
                str(int(value)),
                ha="center",
            )

    fig.suptitle("Destrieux Atlas Region Electrodes", fontsize=16, y=0.97)
    return fig


def plot_electrodes(electrodes: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig = create_electrode_region_figure(electrodes)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nilearn_data_dir = args.nilearn_data_dir or args.output_dir / "nilearn_data"

    region_groups = load_region_groups(args.region_groups_json)
    if not region_groups:
        print(
            "REGION_GROUPS is empty; plotting localized electrodes as unassigned. "
            "Populate utils/atlas_utils.py or pass --region-groups-json for grouped counts."
        )

    rel_paths = discover_data_paths(args.data_root)
    download_missing_files(args.data_root, rel_paths, args.force_download)

    electrodes = load_electrodes(args.data_root, include_bad=args.include_bad)
    electrodes = assign_region_groups(electrodes, region_groups, nilearn_data_dir=nilearn_data_dir)
    assignments_path, counts_path = write_summaries(electrodes, args.output_dir)

    plot_path = args.output_dir / "atlas_region_electrodes.png"
    plot_electrodes(electrodes, plot_path, dpi=args.dpi)

    n_assigned = int((electrodes["region_group"] != "unassigned").sum())
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote assignments: {assignments_path}")
    print(f"Wrote counts: {counts_path}")
    print(f"Localized electrodes: {len(electrodes)}")
    print(f"Assigned to region groups: {n_assigned}")


if __name__ == "__main__":
    main()
