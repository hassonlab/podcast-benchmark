#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output='./logs/%x.out'
#SBATCH --error='./logs/%x.err'

module purge
source decoding_env/bin/activate

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Start time:' `date`
start=$(date +%s)

python "$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
