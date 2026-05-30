#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output='./logs/%x.out'
#SBATCH --error='./logs/%x.err'

module purge
source decoding_env/bin/activate

CHUNK_CACHE_PARENT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CHUNK_CACHE_DIR="$(mktemp -d "${CHUNK_CACHE_PARENT%/}/podcast-chunks-${SLURM_JOB_ID:-$$}.XXXXXX")"

cleanup_chunk_cache() {
    rm -rf "$CHUNK_CACHE_DIR"
}

trap cleanup_chunk_cache EXIT
trap 'cleanup_chunk_cache; exit 143' TERM
trap 'cleanup_chunk_cache; exit 130' INT

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Chunk cache:' "$CHUNK_CACHE_DIR"
echo 'Start time:' `date`
start=$(date +%s)

python "$@" --task_config.data_params.chunked_preprocessing.cache_dir="$CHUNK_CACHE_DIR"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
