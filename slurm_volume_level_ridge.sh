#!/bin/bash
#SBATCH --job-name=vol_lvl_ridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

module purge
module load anaconda3/2025.6
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate decoding_env

cd /scratch/gpfs/HASSON/gidon/podcast-benchmark
mkdir -p logs results models event_logs

CONFIG=configs/volume_level/volume_level_ridge.yml

python main.py \
  --config "${CONFIG}" \
  --model_params.analysis_modes='["pooled_electrodes","per_subject","average"]' \
  --data_params.subject_ids='[1,2,3,4,5,6,7,8,9]'
