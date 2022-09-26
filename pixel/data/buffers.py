from typing import Tuple, List, Dict, Deque
from collections import deque
import random

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
        # idxs = np.random.randint(0, self.size, size=batch_size)
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = dict(observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        return batch

    def __len__(self) -> int:
        return self.size


class NSRBuffer:
    "Simple Replay Buffer for Discrete Action Space (Numpy)"
    # def __init__(self, obs_dim: int, act_dim: int, max_size: int, batch_size: int = 32, seed=0, device='cpu'):
    def __init__(
        self,
        obs_dim: int,
        max_size: int,
        batch_size: int = 32,
        n_steps: int = 1,
        gamma: float = 0.99,
        seed = 0,
        device = 'cpu'):
        self.obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.rew_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.obs_next_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.ter_buf = np.zeros([max_size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size, self.batch_size = 0, 0, max_size, batch_size
        self.n_steps, self.n_steps_buffer = n_steps, deque(maxlen=n_steps)
        self.gamma = gamma
        self._device_ = device

    def store_sarsd(
        self,
        o: np.ndarray,
        a: np.ndarray,
        r: float,
        o_next: np.ndarray,
        d: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:

        sarsd = (o, a, r, o_next, d)
        self.n_steps_buffer.append(sarsd)
        if len(self.n_steps_buffer) < self.n_steps:
            return ()

        o, a = self.n_steps_buffer[0][:2]
        r, o_next, d = self._get_n_steps_info(self.n_steps_buffer, self.gamma)

        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs_next_buf[self.ptr] = o_next
        self.ter_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        return self.n_steps_buffer[0]

    def sample_batch(self, batch_size=32, device='cpu') -> Dict[str, np.ndarray]:
        # idxs = np.random.randint(0, self.size, size=batch_size)
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = dict(idxs=idxs,
                     observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        return batch

    def sample_batch_from_idxs(self, idxs: np.ndarray, device='cpu') -> Dict[str, np.ndarray]:
        batch = dict(observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        return batch

    def _get_n_steps_info(
        self,
        n_steps_buffer: Deque,
        gamma: float,
        ) -> Tuple[float, np.ndarray, bool]:
        reward, observation_next, terminal = self.n_steps_buffer[-1][-3:]
        for sards in reversed(list(n_steps_buffer)[:-1]):
            r, o_next, d = sards[-3:]
            reward = r + gamma*(1-d)*reward
            observation_next, terminal = (o_next, d) if d else (observation_next, terminal)
        return reward, observation_next, terminal

    def __len__(self) -> int:
        return self.size


from .segment_tree import SumSegmentTree, MinSegmentTree

class PERBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer (PER) [ DeepMind(Shaul et al.); ICLR 2016 ]
    """
    def __init__(
        self,
        obs_dim: int,
        max_size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        seed = 0,
        device = 'cpu'):
        assert alpha >= 0
        super(PERBuffer, self).__init__(obs_dim, max_size, batch_size)
        self.alpha, self.max_priority, self.tree_ptr = alpha, 1.0, 0
        tree_capacity = 1
        while tree_capacity < self.max_size: tree_capacity *= 2
        self.sum_tree, self.min_tree = SumSegmentTree(tree_capacity), MinSegmentTree(tree_capacity)

    def store_sarsd(
        self,
        o: np.ndarray,
        a: np.ndarray,
        r: float,
        o_next: np.ndarray,
        d: bool) -> None:
        super().store_sarsd(o, a, r, o_next, d)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr+1) % self.max_size

    def sample_batch(self, batch_size: int = 32, beta: float = 0.4, device='cpu') -> Dict[str, np.ndarray]:
        idxs = self._sample_proportional_idxs(batch_size)
        batch = dict(idxs=idxs,
                     observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs],
                     importance_ws=np.array([self._calc_importance_w(i, beta) for i in idxs]))
        return batch

    def update_prios(self, idxs: List[int], prios: np.ndarray):
        "Update prios of sampled transitions"
        assert len(idxs) == len(prios)
        for i, p in zip(idxs, prios):
            assert p > 0
            assert 0 <= i <= len(self)
            self.sum_tree[i] = p ** self.alpha
            self.min_tree[i] = p ** self.alpha
            self.max_priority = max(self.max_priority, p)

    def _sample_proportional_idxs(self, batch_size: int = 32) -> List[int]:
        "Sample idxs based on proportions"
        idxs = []
        p_total = self.sum_tree.sum(0, len(self)-1)
        segment = p_total / batch_size
        for bs in range(self.batch_size):
            a = segment*bs
            b = segment*(bs+1)
            up_bound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(up_bound)
            idxs.append(idx)
        return idxs

    def _calc_importance_w(self, idx: int, beta: float) -> float:
        "Calculate importance sampling weights at idx"
        p_sample = self.sum_tree[idx]/self.sum_tree.sum()
        importance_w = (p_sample * len(self)) ** (-beta)
        p_min = self.min_tree.min()/self.sum_tree.sum()
        max_w = (p_min * len(self)) ** (-beta)
        importance_w_normz = importance_w / max_w
        return importance_w_normz











class ERMemory:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))

class PERMemory:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, defaul=1))

    def sample(self, batch_size, prio_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self._get_probs(prio_scale)
        # samples = random.choices(self.buffer, k=sample_size)
        sample_idxs = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_idxs]
        importance_w = self._get_importance(sample_probs[sample_idxs])
        return map(list, zip(*samples)), importance_w

    def set_prios(self, idxs, error, offset=0.1):
        for i, e in zip(idxs, error):
            self.priorities[i] = abs(e) + offset

    def _get_probs(self, prio_scale):
        scaled_prios = np.array(self.priorities) ** prio_scale
        sample_probs = scaled_prios / sum(scaled_prios)
        return sample_probs

    def _get_importance(self, probs):
        importance_w = (1/len(beffer)) * (1/probs)
        return importance_w / max(importance_w)
