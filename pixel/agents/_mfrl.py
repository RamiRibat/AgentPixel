import psutil
import argparse

import numpy as np
import gym

from pixel.envs.make import GymMaker
# from pixel.data.buffers import ReplayBuffer, PERBuffer, NSRBuffer

# from pixel.data.replay0 import ReplayBuffer # all n envs
# from pixel.data.replay1 import ReplayBuffer # all n envs
# from pixel.data.replay2 import ReplayBuffer # one env
# from pixel.data.replay3 import ReplayBuffer # x < n envs
from pixel.data.replay4 import ReplayBuffer # n envs (best)
# from pixel.data.memory import ReplayMemory



class MFRL:
    """
    Model-Free Reinforcement Learning
    """
    def __init__(self, configs, seed, device):
        # print('Initialize MFRL!')
        self.configs = configs
        self.seed = seed
        self._device_ = device

    def _build(self):
        self._set_env()
        self._set_buffer()

    def _set_env(self):
        env_cfgs = self.configs['environment']

        self.learn_env = GymMaker(env_cfgs, eval=False, seed=self.seed, device=self._device_)
        if self.configs['evaluation']['evaluate']:
            self.eval_env = GymMaker(env_cfgs, eval=True, seed=self.seed, device=self._device_)

        self.obs_dim = self.learn_env.observation_dim
        self.act_dim = self.learn_env.action_dim
        # self.max_frames = self.learn_env.spec.max_frames_per_episode

    def _set_buffer(self):
        n_envs = self.configs['environment']['n-envs']
        obs_dim, act_dim = self.obs_dim, self.act_dim
        configs = self.configs['data']
        hyperparameters = self.configs['algorithm']['hyperparameters']
        seed, device = self.seed, self._device_
        # self.buffer = ReplayMemory(int(1e5), device)
        self.buffer = ReplayBuffer(
            n_envs=n_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            configs=configs,
            hyperparameters=hyperparameters,
            seed=seed, device=device)

    def interact(self, observation, Z, L, t, Traj, epsilon=0.001):
        xT = self.configs['learning']['expl-steps']

        if t > xT:
            action = self.agent.get_action(observation)
            # action = self.agent.get_e_greedy_action(observation, epsilon=epsilon)
        else:
            action = self.learn_env.action_space.sample()

        # print('interact.observation: ', observation.shape)
        # print('interact.observation: ', observation.max())

        observation_next, reward, terminated, truncated, info = self.learn_env.step(action)

        # print('interact.observation_next: ', observation_next.shape)

        Z += reward
        L += 1

        self.buffer.append_sard(
            observation,
            action,
            reward,
            terminated)

        observation = observation_next

        if terminated or truncated:
            Z, S, L, Traj = 0, 0, 0, Traj+1
            observation, info = self.learn_env.reset()

        return observation, Z, L, Traj, terminated, truncated

    def interact_vec(self, observation, mask, Z, L, T, Traj, epsilon=0.001):
        n_envs = self.configs['environment']['n-envs']
        xT = self.configs['learning']['expl-steps']
        EP = [ epsilon - 0.01*n for n in range(n_envs) ]

        if T > xT:
            # action = self.agent.get_action(observation)
            action = self.agent.get_e_greedy_action_vec(observation, epsilon=EP)
        else:
            action = self.learn_env.action_space.sample()

        # print('interact.observation: ', observation.shape)
        # print('interact.observation: ', observation.max())
        # print('action: ', action)

        observation_next, reward, terminated, truncated, info = self.learn_env.step(action)

        # print('interact.observation_next: ', observation_next.shape)
        # print('reward: ', reward.shape)
        # print('terminated: ', terminated.shape)

        # print('1.observation: ', observation.sum())

        if self.configs['environment']['n-envs'] == 0:
            observation = np.array([observation], dtype=np.float32)
            action = np.array([action], dtype=np.int32)
            reward = np.array([reward], dtype=np.float32)
            terminated = np.array([terminated], dtype=bool)
            truncated = np.array([truncated], dtype=bool)

        # print('2.observation: ', observation.sum())

        self.buffer.append_sard_vec(observation, action, reward, terminated, mask)

        Z += reward[mask].sum() / max(1, n_envs)
        L += mask.sum() / max(1, n_envs)
        steps = mask.sum()

        observation = observation_next
        mask[mask] = ~terminated[mask]
        mask[mask] = ~truncated[mask]

        # if mask.sum()==0:
        if mask.sum()<max(1, n_envs):
            # print(f'RESET | terminated={terminated}')
            Z, S, L, Traj = 0, 0, 0, Traj+max(1, n_envs)
            observation, info = self.learn_env.reset()
            mask = np.ones([max(1, n_envs)], dtype=bool)

        return observation, mask, Z, L, Traj, steps

    def evaluate(self, EE=0):
        evaluate = self.configs['evaluation']['evaluate']
        if evaluate:
            # print('\n[ Evaluation ] -->')
            EE = self.configs['evaluation']['episodes']
            MAX_H = None
            VZ, VS, VL = [], [], []
            for ee in range(1, EE+1):
                # seed = self.seed + np.random.randint(1, 10000)
                Z, S, L = 0, 0, 0
                observation, info = self.eval_env.reset()
                while True:
                    # print(f'[ Evaluation ] ee={ee} | L={L}')
                    # action = self.learn_env.action_space.sample()
                    # action = self.agent.get_greedy_action(observation, evaluation=True)
                    action = self.agent.get_e_greedy_action(observation, evaluation=True)
                    observation, reward, terminated, truncated, info = self.eval_env.step(action)
                    Z += reward
                    L += 1
                    if terminated or truncated: break
                VZ.append(Z)
                VL.append(L)
            self.eval_env.close()
            # print('<-- [ Evaluation ]')
        return VZ, VS, VL
