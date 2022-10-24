from copy import deepcopy
from multiprocessing import Process, Pool
import os, sys, subprocess#, multiprocessing
import argparse
import importlib
import time
import json

import numpy as np
import torch as T

import wandb

import warnings
warnings.filterwarnings('ignore')


valid_algorithms = {'DQN': 'dqn', 'Rainbow': 'rainbow', 'DERainbow': 'rainbow'}


def find_agent(alg):
    alg = valid_algorithms[alg]
    cwd = os.getcwd()
    agent_dir = cwd + '/pixel/agents/'
    for root, dirs, files in os.walk(agent_dir):
        for f in files:
            # if f == (alg.lower() + '.py'):
            if f == (alg + '.py'):
                agent = os.path.join(root, f)
    return agent

def experiment_grid(args):
    exp_grid = []
    info = {}
    for alg in args.alg:
        agent = find_agent(alg)
        configs = args.task.lower() + '_' + alg.lower()
        info['alg'] = alg
        for env in args.env:
            info['env'] = env
            for n in args.n_envs:
                info['n'] = n
                exp_grid.append([agent, configs, deepcopy(info)])
    return exp_grid




def work(processes_list):
    return subprocess.run(processes_list)

def process_work(seeds, processes_list):
    pool = Pool(processes=len(seeds)+1)
    pool.map(work, processes_list)






def main(external_args):
    exp_grid = experiment_grid(external_args)
    for (agent, configs, info) in exp_grid:
        print('=' * 50)
        print(f'Start of an RL experiment')
        print(f"\t Algorithm:   {info['alg']}")
        print(f"\t Environment: {info['env']} X {info['n']}")
        for seed in external_args.seed:
            print(f"\t Random seed: {seed}")
            print('=' * 50)

            subprocess.run(['python', agent,
                            '--env', info['env'],
                            '--n-env', str(info['n']),
                            '--configs', str(configs),
                            '--seed', str(seed),
                            '--device', external_args.device,
                            '--wb', str(external_args.wb) ])

        print('\n')
        print('End of the RL experiment')
        print('=' * 50)


# def main2(external_args):
#     exp_grid = experiment_grid(external_args)
#     for (agent, configs, info) in exp_grid:
#         seeds = external_args.seed
#         device = external_args.device
#         wb = external_args.wb
#         print('=' * 50)
#         print(f'Start of an RL experiment')
#         print(f"\t Algorithm:   {info['alg']}")
#         print(f"\t Environment: {info['env']} X {info['n']}")
#         print(f"\t Random seed(s): {seeds}")
#         print('=' * 50)
#         works_vars = [ [agent, configs, info, seed, device, wb] for seed in seeds]
#
#         work_processes = [ [
#             'python', agent,
#             '--configs', str(configs),
#             '--env', info['env'],
#             '--n-env', str(info['n']),
#             '--seed', str(seed),
#             '--device', device,
#             '--wb', str(wb)
#             ] for seed in seeds ]
#         monitor_process = ['python',  os.getcwd() + '/pixel/utils/monitor.py']
#         work_processes.append(monitor_process)
#
#         process_work(seeds, work_processes)
#
#             # subprocess.run(['python', agent,
#             #                 '--env', info['env'],
#             #                 '--n-env', str(info['n']),
#             #                 '--configs', str(configs),
#             #                 '--seed', str(seed),
#             #                 '--device', external_args.device,
#             #                 '--wb', str(external_args.wb) ])
#
#         print('\n')
#         print('End of the RL experiment')
#         print('=' * 50)
#



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='Atari')
    parser.add_argument('--alg', type=str, nargs='+', default=['Rainbow'])
    parser.add_argument('--env', type=str, nargs='+', default=['ALE/Pong-v5'])
    parser.add_argument('--n-envs', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--wb', action='store_true')

    external_args = parser.parse_args()

    main(external_args)
