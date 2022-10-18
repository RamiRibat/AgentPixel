#!/bin/bash

echo "RL Experiments w/ Bash"

python -m pixel.run -cfg cartpole_dqn -seed 1 -wb
python -m pixel.run -cfg cartpole_dqn -seed 2 -wb
python -m pixel.run -cfg cartpole_dqn -seed 3 -wb
