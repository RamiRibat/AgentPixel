#!/bin/bash
#SBATCH --job-name=atrai_vector

#SBATCH --partition=p100,t4v1,t4v2,rtx6000

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=1G

# prepare your environment here
module load cuda-11.3

# put your command here
conda activate pixel

python -m pixel.run --env "ALE/Pong-v5" --n-envs 2 --device 'cuda' --seed 1 --wb
