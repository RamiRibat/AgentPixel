import psutil
import argparse

import numpy as np
import gym

from pixel.envs.make import GymMaker
from pixel.data.memory import PixelPER
# from pixel.data.buffers import ReplayBuffer, PERBuffer, NSRBuffer





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

        self.learn_env = GymMaker(env_cfgs)
        if self.configs['evaluation']['evaluate']:
            self.eval_env = GymMaker(env_cfgs, eval=True)

        self.obs_dim = self.learn_env.observation_dim
        self.act_dim = self.learn_env.action_dim
        # self.max_frames = self.learn_env.spec.max_frames_per_episode

    def _set_buffer(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        buffer_cfgs = self.configs['data']
        # buffer_size = self.configs['data']['buffer-size']
        # batch_size = self.configs['data']['batch-size']
        buffer_type = self.configs['data']['buffer-type']
        hyper_para = self.configs['algorithm']['hyper-parameters']
        if buffer_type == 'simple':
            self.buffer = ReplayBuffer(obs_dim, max_size, batch_size)
        elif buffer_type == 'per':
            alpha = self.configs['algorithm']['hyper-parameters']['alpha']
            self.buffer = PERBuffer(obs_dim, max_size, batch_size, alpha)
        elif buffer_type == 'simple+nSteps':
            n_steps = self.configs['algorithm']['hyper-parameters']['n-steps']
            self.buffer = NSRBuffer(obs_dim, max_size, batch_size, n_steps=1)
            self.buffer_n = NSRBuffer(obs_dim, max_size, batch_size, n_steps=n_steps)
        elif buffer_type == 'per+nSteps': # Rainbow
            alpha = self.configs['algorithm']['hyper-parameters']['alpha']
            n_steps = self.configs['algorithm']['hyper-parameters']['n-steps']
            self.buffer_per = PERBuffer(obs_dim, max_size, batch_size, alpha)
            self.buffer_n = NSRBuffer(obs_dim, max_size, batch_size, n_steps=n_steps)
        elif buffer_type == 'pixel-simple': # DQN
            self.buffer = PixelER(buffer_cfgs, hyper_para, device=self._device_)
        elif buffer_type == 'pixel-per': # Rainbow
            self.buffer = PixelPER(buffer_cfgs, hyper_para, device=self._device_)

    def interact(self, observation, mask, Z, L, t, Traj, epsilon=0.001):
        n_envs = self.configs['environment']['n-envs']
        n_stacks = self.configs['environment']['n-stacks']
        xT = self.configs['learning']['expl-steps']

        if mask.sum() == 0:
            Z, S, L, Traj = 0, 0, 0, Traj+1
            observation, info = self.learn_env.reset()
            mask = np.ones([max(1, n_envs)], dtype=bool)

        if t > xT:
            action = self.agent.get_action(observation, epsilon=epsilon)
        else:
            action = self.learn_env.action_space.sample()

        # print('observation: ', observation.shape)
        # print('action: ', action)
        observation_next, reward, terminated, truncated, info = self.learn_env.step(action)
        # print('reward: ', reward)
        # print('terminated: ', terminated)

        if self.configs['environment']['n-envs'] == 0:
            observation = np.array([observation])
            action = np.array([action])
            reward = np.array([reward])
            terminated = np.array([terminated])
            truncated = np.array([truncated])

        self.append_sard_in_buffer(
            observation[mask],
            action[mask],
            reward[mask],
            terminated[mask])

        observation = observation_next
        mask[mask] = ~terminated[mask]
        mask[mask] = ~truncated[mask]

        # print('mask: ', mask)

        Z += np.mean(reward)
        L += 1

        return observation, mask, Z, L, Traj

    def interact_vec(self, observation, mask, Z, L, t, Traj, epsilon=0.001):
        n_envs = self.configs['environment']['n-envs']
        n_stacks = self.configs['environment']['n-stacks']
        xT = self.configs['learning']['expl-steps']

        if mask.sum()==0:
            Z, S, L, Traj = 0, 0, 0, Traj+1
            observation, info = self.learn_env.reset()
            mask = np.ones([max(1, n_envs)], dtype=bool)

        if t > xT:
            action = self.agent.get_action(observation, epsilon=epsilon)
        else:
            action = self.learn_env.action_space.sample()

        print('observation: ', observation.shape)
        print('action: ', action)
        observation_next, reward, terminated, truncated, info = self.learn_env.step(action)
        print('reward: ', reward)
        self.append_sard_in_buffer(
            observation[mask*n_stacks],
            action[mask],
            reward[mask],
            terminated[mask])
        observation = observation_next
        mask[mask] = ~terminated[mask]
        mask[mask] = ~truncated[mask]

        Z += reward
        L += 1

        return observation, mask, Z, L, Traj

    def append_sard_in_buffer(
        self,
        observation,
        action,
        reward,
        terminated):

        buffer_type = self.configs['data']['buffer-type']
        if (buffer_type == 'simple') or (buffer_type == 'per'):
            self.buffer.store_sarsd(observation,
                                    action,
                                    reward,
                                    observation_next,
                                    terminated)
        elif buffer_type == 'simple+nSteps':
            self.buffer.store_sarsd(observation,
                                    action,
                                    reward,
                                    observation_next,
                                    terminated)
            self.buffer_n.store_sarsd(observation,
                                      action,
                                      reward,
                                      observation_next,
                                      terminated)
        elif buffer_type == 'per+nSteps':
            self.buffer_per.store_sarsd(observation,
                                    action,
                                    reward,
                                    observation_next,
                                    terminated)
            self.buffer_n.store_sarsd(observation,
                                      action,
                                      reward,
                                      observation_next,
                                      terminated)
        elif buffer_type == 'pixel-per':
            self.buffer.append_sard(observation, action, reward, terminated)

    def evaluate(self):
        evaluate = self.configs['evaluation']['evaluate']
        if evaluate:
            # print('\n[ Evaluation ]')
            EE = self.configs['evaluation']['episodes']
            MAX_H = None
            VZ, VS, VL = [], [], []
            for ee in range(1, EE+1):
                Z, S, L = 0, 0, 0
                observation, info = self.eval_env.reset()
                while True:
                    action = self.agent.get_greedy_action(observation, evaluation=True)
                    # print('observation: ', observation.shape)
                    # print('action: ', action)
                    observation, reward, terminated, truncated, info = self.eval_env.step(action)
                    Z += reward
                    L += 1
                    if terminated or truncated: break
                VZ.append(Z)
                VL.append(L)
        self.eval_env.close()
        return VZ, VS, VL
