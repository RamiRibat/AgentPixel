from typing import Tuple, List, Dict

import numpy as np
import torch as T


class ReplayBuffer:
    "Simple Replay Buffer for Discrete Action Space (Numpy)"
    # def __init__(self, obs_dim: int, act_dim: int, max_size: int, batch_size: int = 32, seed=0, device='cpu'):
    def __init__(self, obs_dim: int, max_size: int, batch_size: int = 32, seed = 0, device = 'cpu'):
        self.obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.rew_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.obs_next_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.ter_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size, self.batch_size = 0, 0, max_size, batch_size
        self._device_ = device

    def store_sarsd(self,
                    o: np.ndarray,
                    a: np.ndarray,
                    r: float,
                    o_next: np.ndarray,
                    d: bool) -> None:
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs_next_buf[self.ptr] = o_next
        self.ter_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, device='cpu') -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        # idxs = np.random.choice(self.size, size=batch_size, repeat=False)
        batch = dict(observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        return batch

    def __len__(self) -> int:
        return self.size
