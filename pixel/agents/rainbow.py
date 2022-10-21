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

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on greedy-policy
        with T.no_grad():
            if evaluation or self.configs['environment']['n-envs']==0:
                return self.online_net(T.tensor(np.array(observation), dtype=T.float32, device=self._device_)).argmax().cpu().numpy()
            else:
                return self.online_net(T.tensor(np.array(observation), dtype=T.float32, device=self._device_)).argmax(1).cpu().numpy()

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
        TUf = self.configs['algorithm']['hyper-parameters']['target-update-frequency']

        oldJq = 0
        Z, S, L, Traj = 0, 0, 0, 0
        RainbowLT = trange(1, LT+1, desc=f'{alg} | s={self.seed}')
        # RainbowLT = trange(1, LT+1, 4, desc=f'{alg} | s={self.seed}')
        observation, info = self.learn_env.reset()
        mask = np.ones([max(1, self.configs['environment']['n-envs'])], dtype=bool)
        total_steps = 0

        logs, ZList, LList, JQList = dict(), [0], [0], []
        SPSList = []
        update_counter, target_counter, logs_counter = (iT//Lf)-1, (iT//TUf)-1, 0

        start_time_real = time.time()

        for t in RainbowLT:
            observation, mask, Z, L, Traj_new = self.interact(observation, mask, Z, L, t, Traj)

            if (Traj_new - Traj) > 0:
                ZList.append(lastZ), LList.append(lastL)
            else:
                lastZ, lastL = Z, L
            Traj = Traj_new

            total_steps += mask.sum()

            # beta = self.update_beta(beta, total_steps, LT)

            # if (total_steps>iT):
            #     self.update_buffer_beta()
            #     if total_steps>(Lf*update_counter):
            #         # for g in range(total_steps//(Lf*update_counter)):
            #         # print(f't={total_steps} | q-update')
            #         Jq, target_counter = self.train_rainbow(total_steps, target_counter)
            #         oldJq = Jq
            #         update_counter += 1
            #     else:
            #         Jq = oldJq
            # else:
            #     Jq = oldJq

            # if ((t-1)%Vf == 0):
            if total_steps>(Vf*logs_counter):
                # print(f't={total_steps} | log+eval')
                cur_time_real = time.time()
                total_time_real = cur_time_real - start_time_real
                sps = total_steps//total_time_real
                SPSList.append(sps)

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
                logs['time/sps                            '] = sps
                logs['time/sps-avg                        '] = np.mean(SPSList)
                RainbowLT.set_postfix({'T': total_steps,'S/S': sps, 'LZ': np.mean(ZList), 'VZ': np.mean(VZ)})
                if self.WandB:
                    wandb.log(logs, step=total_steps)
                logs_counter += 1
            if total_steps >= LT: break


        total_time_real = cur_time_real - start_time_real
        sps = total_steps//total_time_real
        SPSList.append(sps)
        self.agent._evaluation_mode(True)
        VZ, VS, VL = self.evaluate()
        self.agent._evaluation_mode(False)
        logs['data/env_buffer_size                '] = self.buffer._size()
        # logs['training/rainbow/Jq                 '] = Jq # np.mean(JQList)
        logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
        logs['learning/real/rollout_return_std    '] = np.std(ZList)
        logs['learning/real/rollout_length        '] = np.mean(LList)
        logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
        logs['evaluation/episodic_return_std      '] = np.std(VZ)
        logs['evaluation/episodic_length_mean     '] = np.mean(VL)
        logs['time/sps                            '] = sps
        logs['time/sps-avg                        '] = np.mean(SPSList)
        if self.WandB:
            wandb.log(logs, step=total_steps)

        self.learn_env.close()
        self.eval_env.close()

    def train_rainbow(
        self,
        total_steps: int,
        target_counter: float) -> T.Tensor:

        batch_size = self.configs['data']['batch-size']
        TUf = self.configs['algorithm']['hyper-parameters']['target-update-frequency']

        batch = self.buffer.sample_batch(batch_size)
        Jq = self.update_online_net(batch)
        Jq = Jq.item()
        if total_steps>(TUf*target_counter):
            # print(f't={total_steps} > {TUf*target_counter} | target-update')
            self.update_target_net()
            target_counter += 1

        self.agent.online_net.reset_noise()
        self.agent.target_net.reset_noise()
        return Jq, target_counter

    def update_online_net(
        self,
        batch: Dict[str, np.ndarray]) -> T.Tensor:

        norm_clip = self.configs['critic']['network']['optimizer']['norm-clip']
        prio_eps = self.configs['algorithm']['hyper-parameters']['prio-eps']

        idxs = batch['tree_idxs']
        importance_ws = batch['importance_ws']

        Jq_biased = self.compute_Jq_rainbow(batch)
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
        n_steps = self.configs['algorithm']['hyper-parameters']['n-steps']
        atom_size = self.configs['algorithm']['hyper-parameters']['atom-size']
        v_min = self.configs['algorithm']['hyper-parameters']['v-min']
        v_max = self.configs['algorithm']['hyper-parameters']['v-max']
        batch_size = self.configs['data']['batch-size']

        gamma_n = gamma ** n_steps

        observations = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        observations_next = batch['observations_next']
        terminals = batch['terminals']
        # print('observations: ', observations.shape)
        # print('actions: ', actions.shape)
        # print('returns: ', returns.shape)
        # print('terminals: ', terminals.shape)
        # print('support: ', self.agent.online_net.support.shape)

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

    def update_target_net(self) -> None:
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def update_beta(self, beta, t, LT):
        fraction = min(t/LT, 1.0)
        beta = beta + fraction * (1.0 - beta)
        return beta

    def update_buffer_beta(self):
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        beta_i = self.configs['algorithm']['hyper-parameters']['beta']
        beta = self.buffer.beta
        beta_increase = (1-beta_i) / (LT-iT)
        self.buffer.beta = min(beta + beta_increase, 1)

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
