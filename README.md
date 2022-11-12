# Agent Pixel

## RL Implementation for Discrete-Controlled & Pixel-based Environments

## Algorithms
Algorithms we are re-implementing/plannning to re-implement:

| Agents | Classic | Atari | MuJoCo | Distributed |
| --- | :---: | :---: | :---: | :---: |
| DQN | ✅ | ✅ |  | ☑️ |
| Double DQN | ✅ |  |  |  |
| PER-DQN | ✅ |  |  |  |
| Dueling DQN |  |  |  |  |
| A3C |  |  |  |  |
| C51 |  |  |  |  |
| Noisy DQN |  |  |  |  |
| Rainbow | ✅ | ✅ |  |  |
| R2D2 |  |  |  | ☑️ |
| DERainbow |  | ✅ |  |  |
| NGU |  |  |  | ☑️ |
| Agnet57 |  |  |  | ☑️ |

## Selected Results
### Atari 100k/200k DERainbow
| Game | 100k | 200k | 200k x1 | 200k x8 | 200k x64 |
| --- | :---: | :---: | :---: | :---: | :---: |
| Alien | 912 ±338 |  |  |  |  |
| Hero | 27.9 ±3 |  |  |  |  |
| Freeway | 6815 ±1005 |  |  |  |  |
| Pong | -18.3 ±4 |  |  |  |  |
| Qbert | 772 ±364 |  |  |  |  |

### Atari 200M Rainbow x64
| Game | 2M | 5M | 10M | 25M | 50M |
| --- | :---: | :---: | :---: | :---: | :---: |
| Alien |  |  |  |  |  |
| Asterisk |  |  |  |  |  |
| Boxing |  |  |  |  |  |
| Breakout |  |  |  |  |  |
| Hero |  |  |  |  |  |
| Freeway |  |  |  |  |  |
| Pong |  |  |  |  |  |
| Qbert |  |  |  |  |  |

### Atari 200M x64
| Game | DQN | DDQN | PER | Rainbow | ? |
| --- | :---: | :---: | :---: | :---: | :---: |
| Alien |  |  |  |  |  |
| Asterisk |  |  |  |  |  |
| Boxing |  |  |  |  |  |
| Breakout |  |  |  |  |  |
| Hero |  |  |  |  |  |
| Freeway |  |  |  |  |  |
| Pong |  |  |  |  |  |
| Qbert |  |  |  |  |  |

## How to use this code
### Installation (Linux Ubuntu/Debian)
```
conda create -n pixel
pip install -e .
pip install numpy tqdm wandb
pip install opencv-python ale-py gym[accept-rom-license]
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Installation (MacOS)
```
conda create -n pixel
pip install -e .
pip install numpy tqdm wandb
pip install opencv-python ale-py gym[accept-rom-license]
pip install torch
```

### Running Experiments
You can find full default configurations in [pixel/configs](https://github.com/RamiSketcher/AgentPixel/tree/main/pixel/configs), but you can use a few external arguments.
```
conda activate pixel
python -m pixel.run --alg DERainbow --env ALE/Freeway-v5 --n-envs 0 --device 'cuda' --wb --seed 1 2 3
```
* ```--alg``` is the algorithm's name [DQN, DDQN, PER, Rainbow, DERainbow]
* ```--env``` is for environment's id [e.g. Alien-v4, ALE/Alien-v5]
* ```--n-envs``` is for number of envs (0 (default): single-non-vectorized setting, 1+: vectorized setting)
* ```--device``` is for device used for networks training (default: 'cpu')
* ```--wb``` is for activating W&B (default: False)
* ```--seed``` is for random seed(s), one or more (default: 0)


## Acknowledgement
This repo is adapted from [AMMI-RL](https://github.com/RamiSketcher/AMMI-RL), and many other great repos, mostly the following ones (not necessarily in order):
- [OpenAI Gym](https://github.com/openai/gym)
- [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [Rainbow](https://github.com/Kaixhin/Rainbow)
- [rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need/)

## References 

[1] [Human-Level Control Through Deep RL. Mnih et al. @ Nature 2015](https://www.nature.com/articles/nature14236)  
[2] [Deep RL with Double Q-learning. van Hasselt et al. @ AAAI 2016](https://arxiv.org/abs/1509.06461)  
[3] [Prioritized Experience Replay. Schaul et al. @ ICLR 2016](https://arxiv.org/abs/1511.05952?context=cs)  
[4] [Dueling Network Architectures for Deep RLg. Wang et al. @ ICLR 2016](https://arxiv.org/abs/1511.06581)  
[5] [Asynchronous Methods for Deep RL. Mnih et al. @ ICML 2016](https://arxiv.org/abs/1602.01783)  
[6] [A Distributional Perspective on RL. Bellemare et al. @ ICML 2017](https://arxiv.org/abs/1707.06887)  
[7] [Noisy Networks for Exploration. Fortunato et al. @ ICLR 2018](https://arxiv.org/abs/1706.10295)  
[8] [Rainbow: Combining Improvements in Deep RL. Hessel et al. @ AAAI 2018](https://arxiv.org/abs/1710.02298)  
[9] [Recurrent Experience Replay in Distributed RL. Kapturowski et al. @ ICLR 2019](https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning)  
[10] [When to use parametric models in reinforcement learning? van Hasselt et al. @ NeurIPS 2019](https://arxiv.org/abs/1906.05243)  
[11] [Never Give Up: Learning Directed Exploration Strategies. Badia et al. @ ICLR 2020](https://arxiv.org/abs/2002.06038)  
[12] [Agent57: Outperforming the human Atari benchmark. Badia et al. @ PMLR 2020](https://arxiv.org/abs/2003.13350)  
