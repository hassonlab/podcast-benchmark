# Setup Notes for NERSC Perlmutter

## Problems encountered

1. **`setup.sh` fails with `module load anaconda3/2024.6`** — this module doesn't exist on Perlmutter. NERSC uses `module load conda` instead. The `set -e` at the top causes the script to exit immediately on this failure.

2. **`--venv` flag doesn't help** — the `module load` call (line 143-148) runs *before* the conda vs venv branching logic, so `--venv` still hits the error.

3. **Shared conda env `/global/common/software/m4673/decoding_env` has broken PyTorch** — `libtorch_global_deps.so` is missing, causing an `OSError` on `import torch`.

## Solution: create a local venv manually

```bash
cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark

# Data + GloVe should already be downloaded from a prior setup.sh run.
# If not, run: ./setup.sh  (it will fail after downloading data, which is fine)

# Create venv on scratch (avoids home quota issues)
python3 -m venv decoding_env
source decoding_env/bin/activate
pip install -e "."          # base deps only
# pip install -e ".[gpu]"   # if you need bundled CUDA libs (usually not needed on Perlmutter)
```

## Activating later

```bash
source /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark/decoding_env/bin/activate
```

## Running a quick test

```bash
python main.py --config configs/baselines/neural_conv_decoder/glove.yml
```
