import psutil
import argparse
import time, datetime
from tqdm import tqdm, trange

import torch
import numpy as np

import gym
from gym.spaces import Box, Discrete, MultiDiscrete
# from gym.wrappers import AtariPreprocessing, FrameStack
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv

from pixel.envs.wrappers import AtariPreprocessing, FrameStack



class AtariEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        configs,
        eval,
        seed,
        device):

        super().__init__(env)

        # print('Initialize AtariEnv')

        self.configs = configs
        self.eval = eval
        self.seed = seed
        self._device_ = device

        # self._seed()

        self.lives = 0
        self.life_terminated = False

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.ale

    def _seed(self):
        self.env.env.env.seed(self.seed)

    def reset(self, seed=None):
        # print(f'RESET | lives={self.lives} | game-over={self.ale.game_over()}')

        if self.life_terminated and not self.eval:
            # print(f'life-termination')
            self.env.env.life_terminated = False
            # observation, _, _, _, info = self.env.step(0, single_frame = True) # Take 1 NO-OP action only
            observation, _, _, _, info = self.step(0, single_frame = True) # Take 1 NO-OP action only
        else: # eval or train(lives=0)
            # print(f'non-life-termination')
            observation, info = self.env.reset() # Take 30 NO-OP actions
            
        observation = np.asarray(observation, dtype=np.float32)

        # observation, info = self.env.reset() # Take 30 NO-OP actions
        # observation = np.asarray(observation, dtype=np.float32)
        # print(f'reset.observation: {observation.sum(1).sum(1)}')
        # self.env.env.lives = self.env.env.ale.lives()
        return observation, info

    def step(self, action, single_frame = False):
        # return self.env.step(action)
        observation_next, reward, terminated, truncated, info = self.env.step(action, single_frame = False)
        observation_next = np.asarray(observation_next, dtype=np.float32)
        reward_clip = self.configs['reward-clip']
        if reward_clip and not self.eval:
            reward = np.clip(reward, -reward_clip, reward_clip)
        self.life_terminated = self.env.env.life_terminated
        self.lives = self.env.env.lives
        return observation_next, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
