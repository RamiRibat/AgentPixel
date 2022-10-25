#!/bin/bash

module load cuda-11.3

conda activate pixel

python -m pixel.run --env "ALE/Pong-v5" --n-envs 64 --device 'cuda' --wb --seed 1
# python -m pixel.run --env "ALE/Pong-v5" --n-envs 64 --device 'cuda' --wb --seed 2
# python -m pixel.run --env "ALE/Pong-v5" --n-envs 64 --device 'cuda' --wb --seed 3


python -m pixel.run --alg Rainbow --env ALE/Freeway-v5 --device 'cuda' --wb --seed 1




python -m pixel.run --alg DERainbow --env ALE/Freeway-v5 --n-envs 8 --device 'cuda' --wb --seed 1


python -m pixel.run --alg DERainbow --env ALE/Freeway-v5 --device 'cuda' --wb --seed

conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Qbert-v5 --device 'cuda' --wb --seed

conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Hero-v5 --device 'cuda' --wb --seed

conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/RoadRunner-v5 --device 'cuda' --wb --seed



# Version-4
conda activate pixel
python -m pixel.run --alg DERainbow --env Qbert-v4 --device 'cuda' --wb --seed
