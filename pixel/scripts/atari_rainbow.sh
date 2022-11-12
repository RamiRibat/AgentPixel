#!/bin/bash

## Compute Clusters

ssh rahmed@v.vectorinstitute.ai
ssh rahmed@q.vectorinstitute.ai

### Password:

hkhinVIAI22


squeue -u "$USER"

cd GitHub/AgentPixel/

git pull





module load cuda-11.3
srun -c 32 --gres=gpu:1 --mem=16GB --qos=nopreemption -p interactive --pty bash


tmux new -s atari-derainbow-s1
tmux a -t atari-derainbow-s1




conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Alien-v5 --device 'cuda' --wb --seed


conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Freeway-v5 --device 'cuda' --wb --seed


conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Hero-v5 --device 'cuda' --wb --seed


conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Pong-v5 --device 'cuda' --wb --seed


conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Qbert-v5 --device 'cuda' --wb --seed




conda activate pixel
python -m pixel.run --alg DERainbow --env Freeway-v4 --device 'cuda' --wb --seed
