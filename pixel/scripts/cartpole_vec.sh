#!/bin/bash

echo "RL Experiments w/ Bash"

# Vectorized

python -m pixel.run -cfg cartpolex2_dqnvec -seed 1 -wb
python -m pixel.run -cfg cartpolex4_dqnvec -seed 1 -wb
python -m pixel.run -cfg cartpolex8_dqnvec -seed 1 -wb
python -m pixel.run -cfg cartpolex16_dqnvec -seed 1 -wb
