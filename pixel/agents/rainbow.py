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


class RainbowAgent:
    def __init__(self,
                 obs_dim, act_dim,
                 configs, seed, device):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.configs, self.seed = configs, seed
        self._device_ = device
        self.online_net, self.target_net = None, None
        self._build()

    def _build(self):
        net_cfgs = self.configs['critic']['network']
        optimizer = 'T.optim.' + net_cfgs['optimizer']['type']
        lr = net_cfgs['optimizer']['lr']
        eps = net_cfgs['optimizer']['eps']
        self.online_net, self.target_net = self._set_q(), self._set_q()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.online_net.train(), self.target_net.train()
        for p in self.target_net.parameters(): p.requires_grad = False
        self.optimizer = eval(optimizer)(self.online_net.parameters(), lr=lr, eps=eps)

    def _set_q(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        net_cfgs = self.configs['critic']['network']
        hyperparameters  = self.configs['algorithm']['hyperparameters']
        seed, device = self.seed, self._device_
        return NDCQNetwork(obs_dim, act_dim, net_cfgs, hyperparameters, seed, device)

    def get_greedy_action(self, observation, evaluation=False): # Select Action(s) based on greedy-policy
        with T.no_grad():
            observation = T.tensor(observation, dtype=T.float32, device=self._device_)
            if evaluation or self.configs['environment']['n-envs'] == 0:
                _, q_actions = self.online_net(observation.unsqueeze(0)) # [N=1, Stacks, H, W] --> ([A], [1])
                return q_actions.item()#.cpu()#.numpy()
            else:
                _, q_actions = self.online_net(observation) # [N=Xenvs, Stacks, H, W] --> ([N=Xenvs, A], [N=Xenvs])
                return q_actions.cpu().numpy()

    def get_e_greedy_action(self, observation, epsilon=0.001, evaluation=True): # Select Action(s) based on greedy-policy
        if np.random.random() >= epsilon:
            return self.get_greedy_action(observation, evaluation=True)
        else:
            return np.random.randint(0, self.act_dim)

    def get_action(self, observation, evaluation=False): # interaction
        return self.get_greedy_action(observation, evaluation)

    def evaluate_q(self, observation):
        with T.no_grad():
            q_values, _ = self.online_net(observation.unsqueeze(0))
            return q_values.max(1)[0].item()





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
                if (I%Lf==0): self.agent.online_net.reset_noise()

                # observation, Z, L, Traj_new, terminated, truncated = self.interact(observation, Z, L, T, Traj)
                observation, mask, Z, L, Traj_new, steps = self.interact_vec(observation, mask, Z, L, T, Traj)

                if (Traj_new - Traj) > 0:
                    ZList.append(lastZ), LList.append(lastL)
                else:
                    lastZ, lastL = Z, L
                Traj = Traj_new

                # T += 1 # single
                # steps = 1
                T += steps # vec

                RainbowLT.n = T
                RainbowLT.set_postfix({'Traj': Traj, 'LL': lastL, 'LZ': lastZ})
                RainbowLT.refresh()

                if (T>iT): # Start training after iT
                    self.update_buffer_beta(steps)
                    # self.update_buffer_beta2(steps)
                    if (I%Lf==0):
                        for g in range(G):
                            Jq = self.train_rainbow(I)
                        oldJq = Jq
                else:
                    Jq = oldJq
                # Jq = 0

                if (I%Vf==0):
                    RainbowLT.colour = 'MAGENTA'
                    RainbowLT.refresh()
                    cur_time_real = time.time()
                    total_time_real = cur_time_real - start_time_real
                    sps = T/total_time_real
                    SPSList.append(sps)
                    self.agent.online_net.eval()
                    VZ, VS, VL = self.evaluate()
                    self.agent.online_net.train()
                    logs['data/env_buffer_size                '] = self.buffer.size()
                    logs['training/rainbow/Jq                 '] = Jq
                    logs['training/rainbow/beta               '] = self.buffer.beta
                    # logs['training/rainbow/beta               '] = self.buffer.priority_weight
                    logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
                    logs['learning/real/rollout_return_std    '] = np.std(ZList)
                    logs['learning/real/rollout_length        '] = np.mean(LList)
                    logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
                    logs['evaluation/episodic_return_std      '] = np.std(VZ)
                    logs['evaluation/episodic_length_mean     '] = np.mean(VL)
                    logs['evaluation/return_to_length         '] = np.mean(VZ)/np.mean(VL)
                    logs['time/sps                            '] = sps
                    logs['time/sps-avg                        '] = np.mean(SPSList)
                    logs['time/total-real                     '] = total_time_real
                    # RainbowLT.set_postfix({'ss': sps, 'VL': np.mean(VL), 'VZ': np.mean(VZ)})
                    print(f'[ Evaluation ] VL={round(np.mean(VL), 2):<6} | VZ={round(np.mean(VZ), 2):<6} | X={round(np.mean(VZ)/np.mean(VL), 2):<6}')
                    if self.WandB: wandb.log(logs, step=T)
                    RainbowLT.colour = None
                    RainbowLT.refresh()
                I += 1

        RainbowLT.n = T
        RainbowLT.colour = 'GREEN'
        RainbowLT.refresh()

        total_time_real = cur_time_real - start_time_real
        sps = T/total_time_real
        SPSList.append(sps)
        self.agent.online_net.eval()
        VZ, VS, VL = self.evaluate()
        # self.agent.online_net.train()
        logs['data/env_buffer_size                '] = self.buffer.size()
        logs['training/rainbow/Jq                 '] = Jq
        logs['training/rainbow/beta               '] = self.buffer.beta
        # logs['training/rainbow/beta               '] = self.buffer.priority_weight
        logs['learning/real/rollout_return_mean   '] = np.mean(ZList)
        logs['learning/real/rollout_return_std    '] = np.std(ZList)
        logs['learning/real/rollout_length        '] = np.mean(LList)
        logs['evaluation/episodic_return_mean     '] = np.mean(VZ)
        logs['evaluation/episodic_return_std      '] = np.std(VZ)
        logs['evaluation/episodic_length_mean     '] = np.mean(VL)
        logs['evaluation/return_to_length         '] = np.mean(VZ)/np.mean(VL)
        logs['time/sps                            '] = sps
        logs['time/sps-avg                        '] = np.mean(SPSList)
        logs['time/total-real                     '] = total_time_real
        print(f'[ Evaluation ] VL={round(np.mean(VL), 2):<6} | VZ={round(np.mean(VZ), 2):<6} | X={round(np.mean(VZ)/np.mean(VL), 2):<6}')
        if self.WandB: wandb.log(logs, step=T)

        self.learn_env.close()
        self.eval_env.close()
        RainbowLT.close()

    def train_rainbow(
        self,
        I: int) -> T.Tensor:
        # print('[ Training ]-->')

        batch_size = self.configs['data']['batch-size']
        TUf = self.configs['algorithm']['hyperparameters']['target-update-frequency']

        batch = self.buffer.sample_batch(batch_size)
        Jq = self.update_online_net(batch)
        Jq = Jq.item()

        if ((I)%TUf == 0): self.update_target_net()

        # self.agent.online_net.reset_noise()
        # self.agent.target_net.reset_noise()

        # print('<--[ Training ]')

        return Jq

    def update_online_net(
        self,
        batch: Dict[str, np.ndarray]) -> T.Tensor:

        prio_eps = self.configs['algorithm']['hyperparameters']['prio-eps']
        norm_clip = self.configs['critic']['network']['optimizer']['norm-clip']

        idxs = batch['tree_idxs']
        importance_ws = batch['importance_ws']

        Jq_biased = self.compute_Jq_rainbow(batch)
        Jq = T.mean(importance_ws * Jq_biased)

        self.agent.optimizer.zero_grad()
        Jq.backward()
        clip_grad_norm_(self.agent.online_net.parameters(), norm_clip)
        self.agent.optimizer.step()

        Jq_biased = Jq_biased.detach().cpu().numpy()
        new_prios = Jq_biased # + prio_eps
        self.buffer.update_prios(idxs, new_prios)
        # self.buffer.update_priorities(idxs, new_prios)

        return Jq

    def compute_Jq_rainbow(
        self,
        batch: int):

        batch_size = self.configs['data']['batch-size']

        n_steps = self.configs['algorithm']['hyperparameters']['n-steps']
        atom_size = self.configs['algorithm']['hyperparameters']['atom-size']
        Vmin = self.configs['algorithm']['hyperparameters']['v-min']
        Vmax = self.configs['algorithm']['hyperparameters']['v-max']
        gamma = self.configs['algorithm']['hyperparameters']['gamma']

        observations = batch['observations'] # [N, Stacks, H, W]
        actions = batch['actions'] # [N]
        returns = batch['returns'] # [N]
        observations_next = batch['observations_next'] # [N, Stacks, H, W]
        # terminals = batch['terminals'] # [N]
        nonterminals = batch['nonterminals'] # [N]

        # print('observations: ', observations.shape)
        # print('actions: ', actions.shape)
        # print('returns: ', returns.shape)
        # print('terminals: ', terminals.shape)
        # print('support: ', self.agent.online_net.support.shape)

        log_q_probs = self.agent.online_net.q_probs(observations, actions, log=True)

        delatZ = float(Vmax-Vmin) / (atom_size-1)

        with T.no_grad():
            # Nth obs_next probs
            _, q_actions_next = self.agent.online_net(observations_next) # Error: observations :)
            self.agent.target_net.reset_noise()
            q_probs_next = self.agent.target_net.q_probs(observations_next, q_actions_next)

            Tz = returns.unsqueeze(1)\
                + (nonterminals)\
                * (gamma**n_steps)\
                * self.agent.online_net.support.unsqueeze(0)
            Tz = Tz.clamp(min=Vmin, max=Vmax)
            b = (Tz - Vmin) / delatZ

            # Compute L2 projection onto fixed support z
            lb = b.floor().to(T.int64)
            ub = b.ceil().to(T.int64)
            # Fix disappearing probability mass when l = b = u (b is int) (Error: forgot todo)
            lb[ (ub>0) * (lb==ub) ] -= 1
            ub[ (lb<(atom_size-1)) * (lb==ub) ] += 1

            # Ditribute prob of Tz
            offset = T.linspace(
                0, (batch_size-1)*atom_size, batch_size
            ).unsqueeze(1).expand(batch_size, atom_size).to(actions)

            m = observations.new_zeros(batch_size, atom_size)
            m.view(-1).index_add_(
                0, (lb+offset).view(-1), (q_probs_next*(ub.float()-b)).view(-1) )
            m.view(-1).index_add_(
                0, (ub+offset).view(-1), (q_probs_next*(b-lb.float())).view(-1) )


        Jq = -T.sum(m * log_q_probs, 1)

        return Jq

    def update_target_net(self) -> None:
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def update_buffer_beta(self, steps):
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        beta_i = self.configs['algorithm']['hyperparameters']['beta']
        beta = self.buffer.beta
        fraction = steps / (LT-iT)
        beta_increase = fraction * (1-beta_i)
        self.buffer.beta = min(beta + beta_increase, 1)

    def update_buffer_beta2(self, steps):
        LT = self.configs['learning']['total-steps']
        iT = self.configs['learning']['init-steps']
        beta_i = self.configs['algorithm']['hyperparameters']['beta']
        beta = self.buffer.priority_weight
        fraction = steps / (LT-iT)
        beta_increase = fraction * (1-beta_i)
        self.buffer.priority_weight = min(beta + beta_increase, 1)

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
    LT = configurations['learning']['total-steps']

    # group_name = f"{algorithm}-100k-{environment}-X{n_envs}-28" # H < -2.7
    # group_name = f"{algorithm}-200k-{environment}-X{n_envs}" # H < -2.7
    # group_name = f"{algorithm}-200M-{environment}" # H < -2.7

    if n_envs > 0:
        group_name = f"{algorithm}-400k-{environment}-X{n_envs}-v4"
    else:
        group_name = f"{algorithm}-{environment}"

    exp_prefix = f"seed:{seed}"

    project_name = 'ATARI-100-200K' if LT <= int(400e3) else 'ATARI-50M'

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            # project=f'ATARI',
            # project=f'ATARI-100-200K',
            project=project_name,
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
