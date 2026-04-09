# Shared Space for Cross-Patient iEEG Decoding

Cross-patient neural alignment methods for intracranial EEG (iEEG/ECoG) word decoding from podcast listening data.

This repo implements and compares several strategies for building a **shared neural representation** across patients with different electrode placements, then using that shared space to decode word embeddings from brain activity.

Built on top of the [podcast-benchmark](https://github.com/hassonlab/podcast-benchmark) framework.

## Results

| # | Method | Strategy | Pairwise Accuracy |
|---|--------|----------|--------------------|
| 1 | Baseline (raw 183 electrodes) | -- | 0.710 |
| 2 | Global PCA (k=8) | Reconstruct → Decode | 0.720 |
| 3 | Temporal VAE | Reconstruct → Decode | 0.765 |
| 4 | SRM (k=8) | Encode → Decode | 0.772 |
| 5 | **Temporal VAE** | **Encode → Decode** | **0.781** |

**Reconstruct → Decode**: Denoise the original 183 electrode signals via the shared space model, then feed cleaned signals to the word decoder.

**Encode → Decode**: Bypass reconstruction. Feed the compact shared representation (k=8 virtual electrodes x 10 timesteps) directly to the word decoder.

## Data Setup

This repo uses the [Podcast Dataset](https://openneuro.org/datasets/ds005574) (iEEG recordings from 8 patients listening to a podcast).

1. Download the BIDS-formatted dataset
2. Place it at `data/` in the repo root so the structure looks like:
```
data/
  participants.tsv
  derivatives/
    ecogprep/
      sub-01/
      sub-02/
      ...
      sub-09/
```

The `processed_data/` folder (electrode lists, word onsets) is already included in the repo.

## Installation

```bash
pip install -e .
```

Key dependencies: `torch`, `mne`, `gensim`, `transformers`, `scikit-learn`, `pyyaml`.

## Quick Start (Best Result)

### Step 1: Train the Temporal VAE

```bash
python shared_space/scripts/train_temporal_vae.py \
  --config shared_space/configs/podcast_temporal_vae.yml
```

This trains a multi-patient temporal VAE with cross-reconstruction loss. The checkpoint is saved to `shared_space/checkpoints/temporal_vae_podcast.pt`.

### Step 2: Evaluate (Encode → Decode)

```bash
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_temporal_vae_encode_gpt2.yml
```

This encodes all patients into the shared latent space (8 virtual electrodes x 10 timesteps), averages across patients, and feeds the result directly to the word decoder.

## Running All Experiments

### 1. Baseline (no shared space)

```bash
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_gpt2.yml
```

### 2. Global PCA (k=8) — Reconstruct → Decode

```bash
# Train PCA
python shared_space/scripts/train_global_pca.py \
  --config shared_space/configs/podcast_global_pca_k8.yml

# Evaluate
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_global_pca_k8_gpt2.yml
```

### 3. Temporal VAE — Reconstruct → Decode

```bash
# Train (same checkpoint as experiment 5)
python shared_space/scripts/train_temporal_vae.py \
  --config shared_space/configs/podcast_temporal_vae.yml

# Evaluate
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_temporal_vae_reconstruct_gpt2.yml
```

### 4. SRM (k=8) — Encode → Decode

```bash
# Train SRM
python shared_space/scripts/train_srm.py \
  --config shared_space/configs/podcast_srm.yml

# Evaluate
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_srm_encode_gpt2.yml
```

### 5. Temporal VAE — Encode → Decode (Best)

```bash
# Train (same checkpoint as experiment 3)
python shared_space/scripts/train_temporal_vae.py \
  --config shared_space/configs/podcast_temporal_vae.yml

# Evaluate
python main.py \
  --config configs/neural_conv_decoder/neural_conv_decoder_temporal_vae_encode_gpt2.yml
```

## Architecture

### Temporal VAE (Best Method)

The temporal VAE preserves the time dimension in its latent space and uses cross-reconstruction loss to enforce a truly shared coordinate system across patients.

**Encoder** (one per patient):
```
Input: (batch, n_electrodes_i, 10)
  Conv1d → ReLU → BN → Conv1d → ReLU → BN → Conv1d(1x1)
Output: mu, logvar each (batch, k, 10)    # k=8 shared channels
```

**Decoder** (one per patient):
```
Input: z (batch, k, 10)
  ConvTranspose1d → ReLU → BN → ConvTranspose1d → ReLU → BN → ConvTranspose1d(1x1)
Output: (batch, n_electrodes_i, 10)
```

**Training loss**:
```
L = L_recon + alpha * L_cross + beta * L_kl
```
- `L_recon`: MSE between decoder_i(z_avg) and x_i for each patient
- `L_cross`: MSE between decoder_j(mu_i) and x_j for all i != j pairs (forces shared space alignment)
- `L_kl`: KL divergence on the averaged latent

**Evaluation** (Encode → Decode):
```
mu_avg = mean(encoder_i(x_i))  across all 8 patients  →  shape (k, 10)
```
This `mu_avg` is fed directly to the word decoder as 8 "virtual electrodes" over 10 timesteps.

## Project Structure

```
shared_space/
  vae_pipeline.py              # Registers all preprocessors + config setters
  models/
    patient_temporal_vae.py    # MultiPatientTemporalVAE model
  scripts/
    train_temporal_vae.py      # Train temporal VAE
    train_global_pca.py        # Train global PCA
    train_srm.py               # Train SRM
  configs/
    podcast_temporal_vae.yml   # Temporal VAE training config
    podcast_global_pca_k8.yml  # PCA training config
    podcast_srm.yml            # SRM training config

configs/neural_conv_decoder/   # Evaluation configs (one per experiment)
core/                          # Registry system, config classes
models/neural_conv_decoder/    # Word decoder (PitomModel)
tasks/                         # Word embedding decoding task
utils/                         # Data loading, training loop, etc.
processed_data/                # Electrode lists, word onsets
```
