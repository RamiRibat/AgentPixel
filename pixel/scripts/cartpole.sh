#!/bin/bash

echo "RL Experiments w/ Bash"

# python -m pixel.run -cfg cartpole_dqn -seed 1 -wb
# python -m pixel.run -cfg cartpole_dqn -seed 2 -wb
# python -m pixel.run -cfg cartpole_dqn -seed 3 -wb
#
# python -m pixel.run -cfg cartpole_ddqn -seed 1 -wb
# python -m pixel.run -cfg cartpole_ddqn -seed 2 -wb
# python -m pixel.run -cfg cartpole_ddqn -seed 3 -wb
#
# python -m pixel.run -cfg cartpole_per -seed 1 -wb
# python -m pixel.run -cfg cartpole_per -seed 2 -wb
# python -m pixel.run -cfg cartpole_per -seed 3 -wb

# python -m pixel.run -cfg cartpole_nsdqn -seed 1 -wb
# python -m pixel.run -cfg cartpole_nsdqn -seed 2 -wb
# python -m pixel.run -cfg cartpole_nsdqn -seed 3 -wb

python -m pixel.run -cfg cartpole_rainbow -seed 1 -wb
python -m pixel.run -cfg cartpole_rainbow -seed 2 -wb
python -m pixel.run -cfg cartpole_rainbow -seed 3 -wb

# Vectorized

# python -m pixel.run -cfg cartpole_dqnvec -seed 1 -wb
# python -m pixel.run -cfg cartpole_dqnvec -seed 2 -wb
# python -m pixel.run -cfg cartpole_dqnvec -seed 3 -wb
