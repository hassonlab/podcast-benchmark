#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output='./logs/%x.out'
#SBATCH --error='./logs/%x.err'

module purge
module load anaconda3/2025.6
source $(conda info --base)/etc/profile.d/conda.sh
conda activate decoding_env

mkdir -p logs
cd /scratch/gpfs/HASSON/gidon/pb/podcast-benchmark

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Start time:' `date`
start=$(date +%s)

python "$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
