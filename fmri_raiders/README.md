# fmri_raiders — Raider fMRI (tutorial 11 style)

This folder runs analyses on the **Raider** dataset (Haxby et al., 2011) using **preprocessed NumPy** arrays (`movie.npy`, `image.npy`, `label.npy`), in the spirit of [BrainIAK tutorial 11 (SRM)](https://brainiak.org/tutorials/11-SRM/).

Two tracks:

1. **BrainIAK tutorial replication** — shared response model (SRM) and inter-subject correlation (ISC) via `run_tutorial.py` (requires optional dependency `brainiak`).
2. **Temporal VAE** — train on Raider movie data, then evaluate **time-segment matching (TSM)** on the movie and **image leave-one-subject-out decoding** (same conventions as the tutorial where applicable).

---

## Environment and dependencies

From the **repository root** (`podcast-benchmark/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

- **VAE train / eval** (PyTorch only):  
  `pip install -e .`
- **SRM tutorial + SRM baselines inside `eval_temporal_vae.py`**:  
  `pip install -e ".[fmri]"`

BrainIAK may need a system **MPI** runtime. On Ubuntu/Debian, if import fails with MPI errors:

```bash
sudo apt install libopenmpi3 openmpi-bin
```

Then open a new shell and retry.

---

## Downloading the data

You need three files in a single directory (default: `data/raider` under the repo root):

| File        | Role |
|------------|------|
| `movie.npy` | Movie watching BOLD, shape `(n_voxels, n_TR_movie, n_subjects)` |
| `image.npy` | Image localizer runs, shape `(n_voxels, n_TR_image, n_subjects)` |
| `label.npy` | Integer label per image TR (same length as time in `image.npy`) |

These match the **BrainIAK condensed Raider** bundle (not the full multi-GB `brainiak_datasets.zip`).

### Option A — Automated download (recommended)

From the repo root, with the venv activated:

```bash
python fmri_raiders/download_raider_data.py
```

This fetches **`raider.zip`** (~32 MB) from [Zenodo record 2598755](https://zenodo.org/records/2598755), unpacks it, and copies `movie.npy`, `image.npy`, and `label.npy` into **`data/raider/`** by default.

Custom output directory:

```bash
python fmri_raiders/download_raider_data.py --out-dir /path/to/raider
```

If you use a non-default path: **`run_tutorial.py`** accepts **`--data-dir`** or defaults from env **`RAIDER_DATA_DIR`**; **`eval_temporal_vae.py`** uses **`--data-dir`**; **`train_temporal_vae.py`** reads **`data_params.raider_dir`** in the YAML config (default `data/raider`).

### Option B — Manual download

1. Open [Zenodo 2598755](https://zenodo.org/records/2598755).
2. Download **`raider.zip`** (not the entire BrainIAK datasets archive unless you intend to).
3. Unzip and locate the folder that contains **`movie.npy`**, **`image.npy`**, and **`label.npy`**.
4. Either copy those three files into `data/raider/`, or point every command at that folder with `--data-dir`.

---

## Running the BrainIAK-style SRM tutorial

Requires `pip install -e ".[fmri]"` and a working BrainIAK/MPI setup.

```bash
python fmri_raiders/run_tutorial.py --data-dir data/raider --out-dir results/raider_srm
```

Adjust `--data-dir` if you stored the `.npy` files elsewhere.

---

## Temporal VAE: train, then evaluate both tasks

Evaluation implements two **tutorial-style** tests:

- **TSM (time-segment matching)** on the **movie**, using a VAE trained on the **first half** of the movie only (`raider_temporal_vae.yml` → `fmri_raiders/checkpoints/raider_temporal_vae.pt`). Half 2 is held out from VAE training but still **encoded** at test time for segment matching.
- **Image LOO** (leave-one-subject-out linear Nu-SVM on patterns over TRs), using a VAE trained on the **full movie** (`raider_temporal_vae_full_movie.yml` → `fmri_raiders/checkpoints/raider_temporal_vae_full_movie.pt`), matching the “full movie” protocol for image runs.

Metrics use the **encoder latent** (mean `μ`, averaged across subjects’ encoders then pooled over the window), not reconstructed BOLD.

### Train both checkpoints

From repo root (default training device is **CUDA**; add `--device cpu` if needed):

```bash
python fmri_raiders/train_temporal_vae.py --config fmri_raiders/configs/raider_temporal_vae.yml
python fmri_raiders/train_temporal_vae.py --config fmri_raiders/configs/raider_temporal_vae_full_movie.yml
```

### Evaluate both tasks (two checkpoints)

```bash
python fmri_raiders/eval_temporal_vae.py \
  --checkpoint-tsm fmri_raiders/checkpoints/raider_temporal_vae.pt \
  --checkpoint-image fmri_raiders/checkpoints/raider_temporal_vae_full_movie.pt \
  --data-dir data/raider \
  --device auto
```

- Use **`--device auto`** (or `cpu`) on machines without a GPU.
- To use **one** weight file for both TSM and image (not the recommended protocol), pass a single **`--checkpoint PATH`** instead of the two flags above (do not mix `--checkpoint` with `--checkpoint-tsm` / `--checkpoint-image`).
- If BrainIAK is unavailable, eval still runs; SRM baselines are skipped unless installed. To force skipping SRM comparison: **`--no-compare-srm`**.

### One-liner: download, train both, evaluate

```bash
python fmri_raiders/download_raider_data.py && \
python fmri_raiders/train_temporal_vae.py --config fmri_raiders/configs/raider_temporal_vae.yml && \
python fmri_raiders/train_temporal_vae.py --config fmri_raiders/configs/raider_temporal_vae_full_movie.yml && \
python fmri_raiders/eval_temporal_vae.py \
  --checkpoint-tsm fmri_raiders/checkpoints/raider_temporal_vae.pt \
  --checkpoint-image fmri_raiders/checkpoints/raider_temporal_vae_full_movie.pt \
  --data-dir data/raider \
  --device auto
```

---

## References

- BrainIAK tutorial 11 (SRM): https://brainiak.org/tutorials/11-SRM/
- Rendered notebook: https://brainiak.org/notebooks/tutorials/html/11-srm.html
- Raider Zenodo (`raider.zip`): https://zenodo.org/records/2598755
