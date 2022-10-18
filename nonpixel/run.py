import os, subprocess, sys
import argparse
import importlib
import datetime
import random
# import pprint
import json

import numpy as np
import torch as T

import wandb

import warnings
warnings.filterwarnings('ignore')


def main(cfg, seed, device, wb):
    sys.path.append("pixel/configs")
    config = importlib.import_module(cfg)
    configurations = config.configurations

    alg_name = configurations['algorithm']['name']
    env_name = configurations['environment']['name']
    env_domain = configurations['environment']['domain']

    group_name = f"{env_domain}-{env_name}"
    exp_prefix = f"{group_name}-{alg_name}-seed:{seed}"

    print('=' * 50)
    print(f'Start of an RL experiment')
    print(f"\t Algorithm:   {alg_name}")
    print(f"\t Environment: {env_name}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)

    # print('Configurations:\n', configurations)
    print('Configurations:\n', json.dumps(configurations, indent=4, sort_keys=False))

    cwd = os.getcwd()
    agent_dir = cwd + '/pixel/agents/'
    # print('alg_name.lower(): ', alg_name.lower())

    for root, dirs, files in os.walk(agent_dir):
        for f in files:
            if f == (alg_name.lower() + '.py'):
                agent = os.path.join(root, f)


    subprocess.run(['python', agent,
                    '-exp_prefix', exp_prefix,
                    '-cfg', cfg,
                    '-seed', str(seed),
                    '-device', device,
                    '-wb', str(wb) ])


    print('\n')
    print('End of the RL experiment')
    print('=' * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-wb', action='store_true')

    args = parser.parse_args()

    cfg = args.cfg
    seed = args.seed
    # device = 'cuda' if args.gpu else 'cpu'
    device = args.device
    wb = args.wb

    # if seed:
    #     print('seeding')
    #     random.seed(seed), np.random.seed(seed), T.manual_seed(seed)


    main(cfg, seed, device, wb)
