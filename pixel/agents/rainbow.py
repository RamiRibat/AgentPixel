import os, subprocess, sys
import time, datetime
import importlib
import argparse
import random
import psutil
import json

from typing import Tuple, List, Dict
from random import sample
from dataclasses import dataclass
from tqdm import tqdm, trange

import wandb

import numpy as np
import torch as T
nn, F = T.nn, T.nn.functional
from torch.nn.utils import clip_grad_norm_

from pixel.agents._mfrl import MFRL
from pixel.networks.value_functions import NDCQNetwork
from pixel.utils.tools import kill_process


class RainbowAgent:
    def __init__(self,
                 obs_dim, act_dim,
                 configs, seed, device):
        self.obs_dim, self.act_dim= obs_dim, act_dim
        self.configs, self.seed = configs, seed
        self._device_ = device
        self.online_net, self.target_net = None, None
        self._build()

    def _build(self):
        self.online_net, self.target_net = self._set_q(), self._set_q()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def _set_q(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        net_cfgs = self.configs['critic']['network']
        hyper_para  = self.configs['algorithm']['hyper-parameters']
        seed, device = self.seed, self._device_
        return NDCQNetwork(obs_dim, act_dim, net_cfgs, hyper_para, seed, device)

    def get_q(self, observation, action):
        return self.online_net(observation).gather(1, action)

    def get_double_q_target(self, observation):
        with T.no_grad():
            return self.target_net(observation).gather(1, self.online_net(observation).argmax(dim=1, keepdim=True))

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on eps_greedy
        with T.no_grad():
            return self.online_net(T.tensor(np.array(observation), dtype=T.float32)).argmax().cpu().numpy()

    def get_action(self, observation, epsilon=None, evaluation=False):
        return self.get_greedy_action(observation, evaluation)

    def _evaluation_mode(self, mode=False):
        self.online_net._evaluation_mode(mode)



class RainbowLearner(MFRL):
    """
    Rainbow [DeepMind (Hessel et al.); 2017]
    """
    def __init__(self, configs, seed, device, wb):
        super(RainbowLearner, self).__init__(configs, seed, device)
        print('Initialize Rainbow Learner')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()

    def _build(self):
        super(RainbowLearner, self)._build()
        self._build_rainbow()

    def _build_rainbow(self):
        self._set_agent()

    def _set_agent(self):
        self.agent = RainbowAgent(self.obs_dim, self.act_dim, self.configs, self.seed, self._device_)

    def learn(self):
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        xT = self.configs['learning']['expl-steps']
        Lf = self.configs['learning']['learn-freq']
        Vf = self.configs['evaluation']['eval-freq']
        alg = self.configs['algorithm']['name']
        beta = self.configs['algorithm']['hyper-parameters']['beta']

        oldJq = 0
        Z, S, L, Traj = 0, 0, 0, 0
        RainbowLT = trange(1, LT+1, desc=f'{alg} | seed={self.seed}')
        observation, info = self.learn_env.reset()
        logs, ZList, LList, JQList = dict(), [0], [0], []

        for t in RainbowLT:
            observation, Z, L, Traj_new = self.interact(observation, Z, L, t, Traj)
            if (Traj_new - Traj) > 0:
                ZList.append(lastZ), LList.append(lastL)
            else:
                lastZ, lastL = Z, L
            Traj = Traj_new

            # beta = self.update_beta(beta, t, LT)

            # if (t>iT) and ((t-1)%Lf == 0):
            #     Jq = self.train_rainbow(t, beta)
            #     oldJq = Jq
            # else:
            #     Jq = oldJq

            if ((t-1)%Vf == 0):
                self.agent._evaluation_mode(True)
                VZ, VS, VL = self.evaluate()
                self.agent._evaluation_mode(False)
                logs['data/env_buffer_size                '] = self.buffer._size()
                # logs['training/rainbow/Jq                 '] = Jq
                logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
                logs['learning/real/rollout_return_std    '] = np.std(ZList)
                logs['learning/real/rollout_length        '] = np.mean(LList)
                logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
                logs['evaluation/episodic_return_std      '] = np.std(VZ)
                logs['evaluation/episodic_length_mean     '] = np.mean(VL)
                RainbowLT.set_postfix({'Traj': Traj, 'learnZ': np.mean(ZList), 'evalZ': np.mean(VZ)})
                if self.WandB: wandb.log(logs, step=t)

        self.agent._evaluation_mode(True)
        VZ, VS, VL = self.evaluate()
        self.agent._evaluation_mode(False)
        logs['data/env_buffer_size                '] = self.buffer._size()
        # logs['training/rainbow/Jq                  '] = np.mean(JQList)
        logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
        logs['learning/real/rollout_return_std    '] = np.std(ZList)
        logs['learning/real/rollout_length        '] = np.mean(LList)
        logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
        logs['evaluation/episodic_return_std      '] = np.std(VZ)
        logs['evaluation/episodic_length_mean     '] = np.mean(VL)
        if self.WandB: wandb.log(logs, step=t)

        self.learn_env.close()
        self.eval_env.close()

    def train_rainbow(
        self,
        t: int,
        beta: float) -> T.Tensor:

        batch_size = self.configs['data']['batch-size']
        TUf = self.configs['algorithm']['hyper-parameters']['target-update-frequency']
        batch_per = self.buffer.sample_batch(batch_size)
        Jq = self.update_online_net(batch)
        Jq = Jq.item()
        if ((t-1)%TUf == 0): self.update_target_net()

        self.agent.online_net.reset_noise()
        self.agent.target_net.reset_noise()
        return Jq

    def update_online_net(
        self,
        batch: Dict[str, np.ndarray]) -> T.Tensor:

        norm_clip = self.configs['critic']['optimizer']['norm-clip']
        prio_eps = self.configs['algorithm']['hyper-parameters']['prio-eps']

        idxs = batch['tree_idxs']
        importance_ws = batch['importance_ws']

        Jq_biased = self.compute_Jq_rainbow(batch, gamma)
        Jq = T.mean(importance_ws * Jq_biased)

        self.agent.online_net.optimizer.zero_grad()
        Jq.backward()
        clip_grad_norm_(self.agent.online_net.parameters(), norm_clip)
        self.agent.online_net.optimizer.step()

        Jq_biased = Jq_biased.detach().cpu().numpy()
        new_prios = Jq_biased + prio_eps
        self.buffer.update_prios(idxs, new_prios)

        return Jq

    def compute_Jq_rainbow(
        self,
        batch: int):
        gamma = self.configs['algorithm']['hyper-parameters']['gamma']
        atom_size = self.configs['algorithm']['hyper-parameters']['atom-size']
        v_min = self.configs['algorithm']['hyper-parameters']['v-min']
        v_max = self.configs['algorithm']['hyper-parameters']['v-max']
        batch_size = self.configs['data']['batch-size']

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        observations_next = batch['observations_next']
        terminals = batch['terminals']

        delatZ = float(v_max-v_min) / (atom_size-1)

        with T.no_grad():
            q_next_actions = self.agent.online_net(observations_next).argmax(1)
            distribution_next = self.agent.target_net.distribution(observations_next)
            distribution_next = distribution_next[range(batch_size), q_next_actions]

            tZ = rewards + gamma*(1-terminals)*self.agent.online_net.support
            tZ = tZ.clamp(min=v_min, max=v_max)
            b = (tZ - v_min) / delatZ
            lb = b.floor().long()
            ub = b.ceil().long()

            offset = (T.linspace(
                0, (batch_size-1)*atom_size, batch_size
            ).long().unsqueeze(1).expand(batch_size, atom_size).to(self._device_))

            distribution_proj = T.zeros(distribution_next.size(), device=self._device_)
            distribution_proj.view(-1).index_add_(
                0, (lb+offset).view(-1), (distribution_next*(ub.float()-b)).view(-1))
            distribution_proj.view(-1).index_add_(
                0, (ub+offset).view(-1), (distribution_next*(b-lb.float())).view(-1))

        distribution = self.agent.online_net.distribution(observations)
        log_p = T.log( distribution[range(batch_size), actions.view(-1)] )
        Jq = -(distribution_proj * log_p).sum(1)

        return Jq


    def update_target_net(self) -> None:
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def update_beta(self, beta, t, LT):
        fraction = min(t/LT, 1.0)
        import json
        beta = beta + fraction * (1.0 - beta)
        return beta

    def func2(self):
        pass





def main(configurations, seed, device, wb):

    print('Start Rainbow experiment')
    # print('Configurations:\n', json.dumps(configurations, indent=4, sort_keys=False))
    # print('\n')

    algorithm = configurations['algorithm']['name']
    environment = configurations['environment']['name']
    domain = configurations['environment']['domain']
    n_envs = configurations['environment']['n-envs']

    group_name = f"{algorithm}-{environment}-X{n_envs}" # H < -2.7
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            project=f'ATARI',
            config=configs
        )

    rainbow_learner = RainbowLearner(configurations, seed, device, wb)

    rainbow_learner.learn()

    # LS = int(1e3)
    # LT = trange(1, LS+1, desc=f'seed={seed}', position=0)
    # for t in LT:
    #     time.sleep(0.05)

    print('\n')
    print('... End Rainbow experiment')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs', type=str)
    parser.add_argument('--env', type=str, default='ALE/Pong-v5')
    parser.add_argument('--n-envs', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--wb', type=str)

    args = parser.parse_args()

    sys.path.append("pixel/configs")

    configs = importlib.import_module(args.configs)
    seed = int(args.seed)
    device = args.device
    wb = eval(args.wb)

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)

    configurations = configs.configurations
    configurations['environment']['name'] = args.env
    configurations['environment']['n-envs'] = args.n_envs

    # LS = int(1e3)
    # LT = trange(1, LS+1, desc=f'seed={seed}', position=0)
    # for t in LT:
    #     time.sleep(0.005)

    main(configurations, seed, device, wb)

    # kill_process('monitor.py')
