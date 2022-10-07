#!/bin/bash

echo "RL Experiments w/ Bash"

# Vectorized

python test_atari.py -num 0 -wb
python test_atari.py -num 1 -wb
python test_atari.py -num 2 -wb
python test_atari.py -num 4 -wb
python test_atari.py -num 8 -wb
python test_atari.py -num 16 -wb
python test_atari.py -num 32 -wb
