import os, subprocess, sys
import argparse
import importlib
import time, datetime
import random

from typing import Tuple, List, Dict
from random import sample
from dataclasses import dataclass
from tqdm import tqdm, trange

import wandb

import numpy as np
import torch as T
nn, F = T.nn, T.nn.functional
from nn.utils import clip_grad_norm_

from pixel.agents._mfrl import MFRL
from pixel.networks.value_functions import NDCQNetwork


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
        atom_size = self.configs['algorithm']['hyper-parameters']['atom-size']
        v-min = self.configs['algorithm']['hyper-parameters']['v-min']
        v-max = self.configs['algorithm']['hyper-parameters']['v-max']
        net_configs = self.configs['critic']['network']
        seed, device = self.seed, self._device_
        # return QNetwork(obs_dim, act_dim, net_configs, seed, device)
        return NDCQNetwork(obs_dim, act_dim, atom_size, v_min, v_max, net_configs, seed, device)

    def get_q(self, observation, action):
        return self.online_net(observation).gather(1, action)

    def get_double_q_target(self, observation):
        with T.no_grad():
            return self.target_net(observation).gather(1, self.online_net(observation).argmax(dim=1, keepdim=True))

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on eps_greedy
        with T.no_grad():
            if evaluation:
                pass
            else:
                pass
            return self.online_net(T.FloatTensor(observation).to(self._device_)).argmax().cpu().numpy()

    def get_action(self, observation, epsilon=None, evaluation=False):
        return self.get_greedy_action(observation, evaluation)



class RainbowLearner(MFRL):
    """
    Rainbow [DeepMind (Hessel et al.); 2017]
    """
    def __init__(self, exp_prefix, configs, seed, device, wb):
        super(RainbowLearner, self).__init__(exp_prefix, configs, seed, device)
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
        LT = self.configs['learning']['steps']
        iT = self.configs['learning']['init_steps']
        xT = self.configs['learning']['expl_steps']
        Lf = self.configs['learning']['frequency']
        Vf = self.configs['evaluation']['frequency']
        G = self.configs['learning']['grad_steps']
        alg = self.configs['algorithm']['name']
        epsilon = self.configs['algorithm']['hyper-parameters']['init-epsilon']
        beta = self.configs['algorithm']['hyper-parameters']['beta']

        oldJq = 0
        Z, S, L, Traj = 0, 0, 0, 0
        RainbowLT = trange(1, LT+1, desc=alg)
        observation, info = self.learn_env.reset()
        logs, ZList, LList, JQList = dict(), [0], [0], []
        # EPS = []

        for t in RainbowLT:
            observation, Z, L, Traj_new = self.interact(observation, Z, L, t, Traj, epsilon)
            if (Traj_new - Traj) > 0:
                ZList.append(lastZ), LList.append(lastL)
            else:
                lastZ, lastL = Z, L
            Traj = Traj_new

            beta = self.update_beta(beta, t, LT)

            if (t>iT) and ((t-1)%Lf == 0):
                Jq = self.train_rainbow(t, beta)
                oldJq = Jq
            else:
                Jq = oldJq

            if ((t-1)%Vf == 0):
                VZ, VS, VL = self.evaluate()
                logs['data/env_buffer_size                '] = self.buffer.size
                logs['training/rainbow/Jq                     '] = Jq
                logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
                logs['learning/real/rollout_return_std    '] = np.std(ZList)
                logs['learning/real/rollout_length        '] = np.mean(LList)
                logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
                logs['evaluation/episodic_return_std      '] = np.std(VZ)
                logs['evaluation/episodic_length_mean     '] = np.mean(VL)
                RainbowLT.set_postfix({'Traj': Traj, 'AvgZ': np.mean(ZList), 'AvgL': np.mean(LList)})
                if self.WandB: wandb.log(logs, step=t)

        VZ, VS, VL = self.evaluate()
        logs['data/env_buffer_size                '] = self.buffer.size
        logs['training/rainbow/Jq                     '] = np.mean(JQList)
        logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
        logs['learning/real/rollout_return_std    '] = np.std(ZList)
        logs['learning/real/rollout_length        '] = np.mean(LList)
        logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
        logs['evaluation/episodic_return_std      '] = np.std(VZ)
        logs['evaluation/episodic_length_mean     '] = np.mean(VL)
        if self.WandB: wandb.log(logs, step=t)

        self.learn_env.close()
        self.eval_env.close()

    def train_rainbow(self, t, beta) -> T.Tensor:
        batch_size = self.configs['data']['batch_size']
        TUf = self.configs['algorithm']['hyper-parameters']['target_update_frequency']
        batch_per = self.buffer_per.sample_batch(batch_size, device=self._device_)
        idxs = batch_per['idxs']
        batch_n = self.buffer_n.sample_batch_from_idxs(idxs, device=self._device_)
        Jq = self.update_online_net(batch_per, batch_n)
        Jq = Jq.item()
        if ((t-1)%TUf == 0): self.update_target_net()

        self.online_net.reset_noise()
        self.target_net.reset_noise()
        return Jq

    def update_online_net(
        self,
        batch_per: Dict[str, np.ndarray],
        batch_n: Dict[str, np.ndarray]) -> T.Tensor:

        prio_eps = self.configs['algorithm']['hyper-parameters']['prio-eps']
        n_steps = self.configs['algorithm']['hyper-parameters']['n-steps']
        gamma = self.configs['algorithm']['hyper-parameters']['gamma']
        gamma_n = gamma ** n_steps

        idxs = batch_per['idxs']
        importance_ws = batch_per['importance_ws']

        Jq_per = self.compute_Jq_per(batch_per, gamma, prio_eps)
        Jq_n = self.compute_Jq_n(batch_n, gamma_n)
        Jq_biased = Jq_per + Jq_n

        Jq = T.mean(importance_ws * Jq_biased)

        self.agent.online_net.optimizer.zero_grad()
        Jq.backward()
        clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.agent.online_net.optimizer.step()

        per_idxs = batch_per['idxs']

        Jq_per = Jq_per.detach().cpu().numpy()
        new_prios = Jq_per + prio_eps
        self.buffer_per.update_prios(per_idxs, new_prios)

        return Jq

    def compute_Jq_per(self, batch, gamma, prio_eps):
        # idxs = batch['idxs']
        observations = T.FloatTensor(batch['observations']).to(self._device_)
        actions = T.LongTensor(batch['actions']).to(self._device_)
        rewards = T.FloatTensor(batch['rewards']).to(self._device_)
        observations_next = T.FloatTensor(batch['observations_next']).to(self._device_)
        terminals = T.FloatTensor(batch['terminals']).to(self._device_)
        # importance_ws = T.FloatTensor(batch['importance_ws']).to(self._device_)

        q_value = self.agent.get_q(observations, actions)
        q_next = self.agent.get_q_target(observations_next)
        q_target = rewards + gamma * (1 - terminals) * q_next
        Jq = F.smooth_l1_loss(q_value, q_target, reduction='none')
        # Jq = T.mean(importance_ws * Jq_biased)

        return Jq

    def compute_Jq_n(self, batch_n, gamma_n):
        observations = T.FloatTensor(batch['observations']).to(self._device_)
        actions = T.LongTensor(batch['actions']).to(self._device_)
        rewards = T.FloatTensor(batch['rewards']).to(self._device_)
        observations_next = T.FloatTensor(batch['observations_next']).to(self._device_)
        terminals = T.FloatTensor(batch['terminals']).to(self._device_)

        q_value = self.agent.get_q(observations, actions)
        q_next = self.agent.get_q_target(observations_next)
        q_target = rewards + gamma_n * (1 - terminals) * q_next
        Jq = F.smooth_l1_loss(q_value, q_target)

        return Jq

    def compute_Jq_rainbow(self, batch):
        atom_size = self.configs['algorithm']['hyper-parameters']['atom-size']
        v_min = self.configs['algorithm']['hyper-parameters']['v-min']
        v_max = self.configs['algorithm']['hyper-parameters']['v-max']

        observations = T.FloatTensor(batch['observations']).to(self._device_)
        actions = T.LongTensor(batch['actions']).to(self._device_)
        rewards = T.FloatTensor(batch['rewards']).to(self._device_)
        observations_next = T.FloatTensor(batch['observations_next']).to(self._device_)
        terminals = T.FloatTensor(batch['terminals']).to(self._device_)

        delatZ = float(v_max-v_min) / (atom_size-1)


    def update_target_net(self) -> None:
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def update_beta(self, beta, t, LT):
        fraction = min(t/LT, 1.0)
        beta = beta + fraction * (1.0 - beta)
        return beta

    def func2(self):
        pass





def main(exp_prefix, config, seed, device, wb):

    print('Start Rainbow experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_domain = configs['environment']['domain']

    group_name = f"{env_name}-{alg_name}-1" # H < -2.7
    exp_prefix = f"seed:{seed}"
    # print('group: ', group_name)

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            project=f'VECTOR',
            config=configs
        )

    rainbow_learner = RainbowLearner(exp_prefix, configs, seed, device, wb)

    # rainbow_learner.learn()

    print('\n')
    print('... End Rainbow experiment')



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-exp_prefix', type=str)
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=str)
    parser.add_argument('-device', type=str)
    parser.add_argument('-wb', type=str)

    args = parser.parse_args()

    exp_prefix = args.exp_prefix
    # sys.path.append(f"{os.getcwd()}/configs")
    sys.path.append(f"pixel/configs")
    config = importlib.import_module(args.cfg)
    seed = int(args.seed)
    device = args.device
    wb = eval(args.wb)

    main(exp_prefix, config, seed, device, wb)
