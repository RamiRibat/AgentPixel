#!/bin/bash

echo "RL Experiments w/ Bash"

python -m pixel.run -cfg atari_rainbow -seed 1 -wb
# python -m pixel.run -cfg atari_rainbow -seed 2 -wb
# python -m pixel.run -cfg atari_rainbow -seed 3 -wb

# Vectorized

# python -m pixel.run -cfg atari_dqnvec -seed 1 -wb
# python -m pixel.run -cfg atari_dqnvec -seed 2 -wb
# python -m pixel.run -cfg atari_dqnvec -seed 3 -wb



python -m pixel.run --env "ALE/Pong-v5" --n-envs 0 --device 'cuda'
