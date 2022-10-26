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
from pixel.networks.value_functions import NDCQNetwork, NDCQNetwork2
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
        hyperparameters  = self.configs['algorithm']['hyperparameters']
        seed, device = self.seed, self._device_
        return NDCQNetwork(obs_dim, act_dim, net_cfgs, hyperparameters, seed, device)

    def get_q(self, observation, action):
        return self.online_net(observation).gather(1, action)

    def get_double_q_target(self, observation):
        with T.no_grad():
            return self.target_net(observation).gather(1, self.online_net(observation).argmax(dim=1, keepdim=True))

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on greedy-policy
        with T.no_grad():
            observation = T.tensor(np.array(observation), dtype=T.float32, device=self._device_)
            if evaluation or self.configs['environment']['n-envs']==0:
                return self.online_net(observation).argmax().cpu().numpy()
            else:
                return self.online_net(observation).argmax(1).cpu().numpy()

    def get_e_greedy_action(self, observation, epsilon=0.001, evaluation=True): # Select Action(s) based on greedy-policy
        with T.no_grad():
            if np.random.random() >= epsilon:
                return self.get_greedy_action(observation, evaluation=True)
            else:
                return np.random.randint(0, self.act_dim)

    def get_action(self, observation, epsilon=None, evaluation=False): # interaction
        return self.get_greedy_action(observation, evaluation)

    def _evaluation_mode(self, mode=False):
        self.online_net._evaluation_mode(mode)



class RainbowAgent2:
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
        hyperparameters  = self.configs['algorithm']['hyperparameters']
        seed, device = self.seed, self._device_
        return NDCQNetwork2(obs_dim, act_dim, net_cfgs, hyperparameters, seed, device)

    def get_q(self, observation, action):
        return self.online_net(observation).gather(1, action)

    def get_double_q_target(self, observation):
        with T.no_grad():
            return self.target_net(observation).gather(1, self.online_net(observation).argmax(dim=1, keepdim=True))

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on greedy-policy
        with T.no_grad():
            observation = T.tensor(np.array(observation), dtype=T.float32, device=self._device_)
            if evaluation or self.configs['environment']['n-envs']==0:
                return (self.online_net(observation.unsqueeze(0)) * self.online_net.support).sum(2).argmax(1).item()
            else:
                return (self.online_net(observation.unsqueeze(0)) * self.online_net.support).sum(2).argmax(1).item()

    def get_e_greedy_action(self, observation, epsilon=0.001, evaluation=True): # Select Action(s) based on greedy-policy
        with T.no_grad():
            if np.random.random() >= epsilon:
                return self.get_greedy_action(observation, evaluation=True)
            else:
                return np.random.randint(0, self.act_dim)

    def get_action(self, observation, epsilon=None, evaluation=False): # interaction
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
        # self.agent = RainbowAgent2(self.obs_dim, self.act_dim, self.configs, self.seed, self._device_)

    def learn(self):
        n_envs = self.configs['environment']['n-envs']
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        xT = self.configs['learning']['expl-steps']
        Lf = self.configs['learning']['learn-freq']# // n_envs
        G =  self.configs['learning']['grad-steps']
        Vf = self.configs['evaluation']['eval-freq']# // n_envs

        alg = self.configs['algorithm']['name']
        beta = self.configs['algorithm']['hyperparameters']['beta']
        # TUf = self.configs['algorithm']['hyperparameters']['target-update-frequency']

        oldJq = 0
        total_steps, g = 0, 0
        Z, S, L, Traj = 0, 0, 0, 0
        RainbowLT = tqdm(total=LT, desc=alg, position=0)
        logs, ZList, LList, JQList = dict(), [0], [0], []
        lastZ, lastL = 0, 0
        # EPS = []
        SPSList = []

        start_time_real = time.time()

        # for t in RainbowLT:
        with RainbowLT:
            observation, info = self.learn_env.reset()
            mask = np.ones([max(1, n_envs)], dtype=bool)
            T, I = 0, 0
            while T<LT:
                # observation, Z, L, Traj_new = self.interact(observation, Z, L, T, Traj)
                observation, mask, Z, L, Traj_new, steps = self.interact_vec(observation, mask, Z, L, T, Traj)

                if (Traj_new - Traj) > 0:
                    ZList.append(lastZ), LList.append(lastL)
                else:
                    lastZ, lastL = Z, L
                Traj = Traj_new

                # T += 1 # single
                T += steps # vec
                # T = self.buffer.size()
                RainbowLT.n = T
                RainbowLT.refresh()

                # if (T>iT): # Start training after iT
                #     self.update_buffer_beta(steps)
                #     if (I%Lf==0):
                #         for g in range(G):
                #             Jq = self.train_rainbow(I)
                #         oldJq = Jq
                # else:
                #     Jq = oldJq
                Jq = 0

                if (I%Vf==0):
                    RainbowLT.colour = 'MAGENTA'
                    RainbowLT.refresh()
                    cur_time_real = time.time()
                    total_time_real = cur_time_real - start_time_real
                    sps = T/total_time_real
                    SPSList.append(sps)
                    self.agent._evaluation_mode(True), self.agent.online_net.eval()
                    VZ, VS, VL = self.evaluate()
                    self.agent._evaluation_mode(False), self.agent.online_net.train()
                    logs['data/env_buffer_size                '] = self.buffer.size()
                    logs['training/rainbow/Jq                 '] = Jq
                    logs['training/rainbow/beta               '] = self.buffer.beta
                    logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
                    logs['learning/real/rollout_return_std    '] = np.std(ZList)
                    logs['learning/real/rollout_length        '] = np.mean(LList)
                    logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
                    logs['evaluation/episodic_return_std      '] = np.std(VZ)
                    logs['evaluation/episodic_length_mean     '] = np.mean(VL)
                    logs['time/sps                            '] = sps
                    logs['time/sps-avg                        '] = np.mean(SPSList)
                    logs['time/total-real                     '] = total_time_real
                    RainbowLT.set_postfix({'ss': sps, 'LZ': lastZ, 'VZ': f'{np.mean(VZ)}'})
                    print(f'VL={np.mean(VL)} | VZ={np.mean(VZ)}')
                    if self.WandB: wandb.log(logs, step=T)
                    RainbowLT.colour = None #'BLACK'
                    RainbowLT.refresh()
                I += 1

        RainbowLT.close()

        total_time_real = cur_time_real - start_time_real
        sps = T/total_time_real
        SPSList.append(sps)
        self.agent._evaluation_mode(True), self.agent.online_net.eval()
        VZ, VS, VL = self.evaluate()
        self.agent._evaluation_mode(False), self.agent.online_net.train()
        logs['data/env_buffer_size                '] = self.buffer.size()
        logs['training/rainbow/Jq                 '] = Jq
        logs['training/rainbow/beta               '] = self.buffer.beta
        logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
        logs['learning/real/rollout_return_std    '] = np.std(ZList)
        logs['learning/real/rollout_length        '] = np.mean(LList)
        logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
        logs['evaluation/episodic_return_std      '] = np.std(VZ)
        logs['evaluation/episodic_length_mean     '] = np.mean(VL)
        logs['time/sps                            '] = sps
        logs['time/sps-avg                        '] = np.mean(SPSList)
        logs['time/total-real                     '] = total_time_real
        if self.WandB: wandb.log(logs, step=T)

        self.learn_env.close()
        self.eval_env.close()

    def train_rainbow(
        self,
        I: int) -> T.Tensor:

        batch_size = self.configs['data']['batch-size']
        TUf = self.configs['algorithm']['hyperparameters']['target-update-frequency']

        batch = self.buffer.sample_batch(batch_size)
        Jq = self.update_online_net(batch)
        Jq = Jq.item()

        if ((I)%TUf == 0): self.update_target_net()

        self.agent.online_net.reset_noise()
        self.agent.target_net.reset_noise()

        return Jq

    def update_online_net(
        self,
        batch: Dict[str, np.ndarray]) -> T.Tensor:

        prio_eps = self.configs['algorithm']['hyperparameters']['prio-eps']

        idxs = batch['tree_idxs']
        importance_ws = batch['importance_ws']

        Jq_biased = self.compute_Jq_rainbow(batch)
        # Jq_biased = self.compute_Jq_rainbow2(batch)
        Jq = T.mean(importance_ws * Jq_biased)

        self.agent.online_net.optimizer.zero_grad()
        Jq.backward()
        clip_grad_norm_(self.agent.online_net.parameters(), 10.0)
        self.agent.online_net.optimizer.step()

        Jq_biased = Jq_biased.detach().cpu().numpy()
        new_prios = Jq_biased + prio_eps
        self.buffer.update_prios(idxs, new_prios)

        return Jq


    def compute_Jq_rainbow(
        self,
        batch: int):

        batch_size = self.configs['data']['batch-size']

        n_steps = self.configs['algorithm']['hyperparameters']['n-steps']
        atom_size = self.configs['algorithm']['hyperparameters']['atom-size']
        v_min = self.configs['algorithm']['hyperparameters']['v-min']
        v_max = self.configs['algorithm']['hyperparameters']['v-max']
        gamma = self.configs['algorithm']['hyperparameters']['gamma']
        gamma_n = gamma ** n_steps

        observations = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        observations_next = batch['observations_next']
        terminals = batch['terminals']

        delatZ = float(v_max-v_min) / (atom_size-1)

        with T.no_grad():
            q_next_actions = self.agent.online_net(observations_next).argmax(1)
            distribution_next = self.agent.target_net.distribution(observations_next)
            distribution_next = distribution_next[range(batch_size), q_next_actions]

            tZ = returns + gamma_n*(1-terminals)*self.agent.online_net.support
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

    def compute_Jq_rainbow2(
        self,
        batch: int):

        batch_size = self.configs['data']['batch-size']

        n_steps = self.configs['algorithm']['hyperparameters']['n-steps']
        atom_size = self.configs['algorithm']['hyperparameters']['atom-size']
        Vmin = self.configs['algorithm']['hyperparameters']['v-min']
        Vmax = self.configs['algorithm']['hyperparameters']['v-max']
        gamma = self.configs['algorithm']['hyperparameters']['gamma']

        observations = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        observations_next = batch['observations_next']
        terminals = batch['terminals']

        log_ps = self.agent.online_net(observations, log=True)
        log_ps_a = log_ps[range(batch_size), actions]

        delatZ = float(Vmax-Vmin) / (atom_size-1)

        with T.no_grad():
            # Nth obs_next probs
            pns = self.agent.online_net(observations)
            dns = self.agent.online_net.support.expand_as(pns) * pns
            argmax_idxs_ns = dns.sum(2).argmax(1)
            self.agent.target_net.reset_noise()
            pns = self.agent.target_net(observations_next)
            pns_a = pns[range(batch_size), argmax_idxs_ns]
            # q_next_actions = self.agent.online_net(observations_next).argmax(1)
            # distribution_next = self.agent.target_net.distribution(observations_next)
            # distribution_next = distribution_next[range(batch_size), q_next_actions]

            # Tz (Belleman op)
            Tz = returns.unsqueeze(1) + (gamma**n_steps) * (1-terminals) * self.agent.online_net.support.unsqueeze(0)
            Tz = Tz.clamp(min=Vmin, max=Vmax)
            # print('Tz: ', Tz.shape)
            b = (Tz - Vmin) / delatZ
            # tZ = returns + gamma_n*(1-terminals)*self.agent.online_net.support
            # tZ = tZ.clamp(min=v_min, max=v_max)
            # b = (tZ - v_min) / delatZ

            # Compute L2 projection onto fixed support z
            lb = b.floor().to(T.int64)
            ub = b.ceil().to(T.int64)
            # lb = b.floor().long()
            # ub = b.ceil().long()
            # Fix prob dissappearing mass when lb = ub = b (b; int)

            # Ditribute prob of Tz
            m = observations.new_zeros(batch_size, atom_size)
            offset = (T.linspace(
                0, (batch_size-1)*atom_size, batch_size
            ).unsqueeze(1).expand(batch_size, atom_size).to(actions))
            m.view(-1).index_add_(
                0, (lb+offset).view(-1), (pns_a*(ub.float()-b)).view(-1))
            m.view(-1).index_add_(
                0, (ub+offset).view(-1), (pns_a*(b-lb.float())).view(-1))
            # distribution_proj = T.zeros(distribution_next.size(), device=self._device_)
            # offset = (T.linspace(
            #     0, (batch_size-1)*atom_size, batch_size
            # ).long().unsqueeze(1).expand(batch_size, atom_size).to(self._device_))
            # distribution_proj.view(-1).index_add_(
            #     0, (lb+offset).view(-1), (distribution_next*(ub.float()-b)).view(-1))
            # distribution_proj.view(-1).index_add_(
            #     0, (ub+offset).view(-1), (distribution_next*(b-lb.float())).view(-1))

        # distribution = self.agent.online_net.distribution(observations)
        # log_p = T.log( distribution[range(batch_size), actions.view(-1)] )

        Jq = -T.sum(m * log_ps_a, 1)

        return Jq

    def update_target_net(self) -> None:
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def update_buffer_beta0(self, T):
        LT = self.configs['learning']['total-steps']
        beta = self.buffer.beta
        fraction = min(T/LT, 1.0)
        self.buffer.beta = beta + fraction * (1.0 - beta)

    def update_buffer_beta(self, steps):
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        beta_i = self.configs['algorithm']['hyperparameters']['beta']
        beta = self.buffer.beta
        fraction = steps / (LT-iT)
        beta_increase = fraction * (1-beta_i)
        self.buffer.beta = min(beta + beta_increase, 1)

    def func2(self):
        pass




def main(configurations, seed, device, wb):

    print('Start Rainbow experiment')
    # print('Configurations:\n', json.dumps(configurations, indent=4, sort_keys=False))
    # print('\n')

    # LT = configurations[][]
    algorithm = configurations['algorithm']['name']
    environment = configurations['environment']['name']
    domain = configurations['environment']['domain']
    n_envs = configurations['environment']['n-envs']

    group_name = f"{algorithm}-100k-{environment}-X{n_envs}" # H < -2.7
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            project=f'ATARI',
            config=configurations
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
        # T.manual_seed(seed) # bad choice
        T.manual_seed(np.random.randint(1, 10000))
        if device == 'cuda':
            # T.cuda.manual_seed(seed) # bad choice
            T.cuda.manual_seed(np.random.randint(1, 10000))
            T.backends.cudnn.enabled = True

    configurations = configs.configurations
    configurations['environment']['name'] = args.env
    configurations['environment']['n-envs'] = args.n_envs

    main(configurations, seed, device, wb)
