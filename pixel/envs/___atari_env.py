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
    def __init__(self, configs, eval=False, device=None, seed=0):
        # print('Initialize GymMaker')
        self.configs = configs
        self.eval = eval
        self._device_ = device
        self.seed = seed
        self.name = configs['name']
        self.env = self._gym_make(configs, eval)

        super().__init__(self.env)

        # self.observation_space = self.env.observation_space
        # if configs['state'] == 'pixel':
        #     self.observation_dim = 'pixel'
        # # else: # ROM
        # #     self.observation_dim = int(np.prod(self.observation_space.shape[0]))
        #
        # self.action_space = self.env.action_space
        #
        # if isinstance(self.action_space, Box):
        #     self.action_dim = self.action_space.shape[0]
        # elif isinstance(self.action_space, Discrete):
        #     self.action_dim = self.action_space.n
        # elif isinstance(self.env.single_action_space, Discrete):
        #     self.action_dim = self.env.single_action_space.n

        # self.seed()
        self.lives = 0
        self.life_terminated = False

    def _gym_make(self, configs, eval):
        def create_env():
            def _make():
                env = gym.make(
                        id=configs['name'],
                        frameskip=configs['frameskip'],
                        max_num_frames_per_episode=configs['max-frames'],
                        repeat_action_probability=configs['repeat-action-probability'],
                        )
                if configs['state'] == 'pixel':
                    env = AtariPreprocessing(
                            env=env,
                            **configs['pre-processing'])
                    env = FrameStack(
                            env=env,
                            num_stack=configs['n-stacks'])
                    pass
                return env
            return _make

        if self.eval:
            if eval: configs['pre-processing']['terminal_on_life_loss'] = False
            env = create_env()
            return env()
        else:
            env = create_env()
            return env()

        # if self.eval:
        #     if eval: configs['pre-processing']['terminal_on_life_loss'] = False
        #     env = create_env()
        #     return env()
        # else:
        #     if configs['n-envs'] == 0:
        #         env = create_env()
        #         return env()
        #     else:
        #         env_fns = [ create_env() for e in range(configs['n-envs']) ]
        #         return AsyncVectorEnv(env_fns) if configs['asynchronous'] else SyncVectorEnv(env_fns)

    def seed(self):
        seed = self.seed
        pass

    def reset(self, seed=0):
        # return self.env.reset()
        # print('AtariEnv.reset')
        if self.life_terminated:
            # print(f'reset-life-terminated: ale.lives={self.env.env.ale.lives()} | life-terminated={self.life_terminated}')
            self.env.env.life_terminated = self.life_terminated = False
            self.env.env.ale.act(0) # Take 1 NO-OP action only
            observation_i = self.env.env._get_obs()
            self.env.frames.append(observation_i)
            info = [] # self.env.env._get_info()
        else: # eval or train(lives=0)
            # print(f'reset-start: ale.lives={self.env.env.ale.lives()} | life-terminated={self.life_terminated}')
            observation, info = self.env.reset()
        observation = np.stack(self.env.observation(None), 0)
        self.lives = self.env.env.ale.lives()
        return observation, info

    def step(self, action = 0):
        # return self.env.step(action)
        # print('atarienv-action: ', action)
        # print(f'AtariEnv.step: lives={self.lives} | ale.lives={self.env.env.ale.lives()} | life-terminated={self.life_terminated}')
        observation_next, reward, terminated, truncated, info = self.env.step(action)
        observation_next = np.stack(observation_next, 0)
        if self.configs['reward-clip'] and not self.eval:
            reward_clip = self.configs['reward-clip']
            reward = np.clip(reward, -reward_clip, reward_clip)
        self.life_terminated = self.env.env.life_terminated
        return observation_next, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
