"""
Source: https://github.com/Kaixhin/Rainbow/blob/master/memory.py
"""
from typing import Dict
import numpy as np
import torch as T

blank_sard = (0, np.zeros((84,84), dtype=np.uint8), 0, 0.0, False)
sard_dtype = np.dtype([
    ('t', np.int32),
    ('observation', np.uint8, (84,84)),
    ('action', np.int32),
    ('reward', np.float32),
    ('terminal', np.bool_),
])


class SegmentTree: # Done
    def __init__(self, capacity):
        self.tree_capacity, self.full = capacity, False
        self.idx, self.tree_start = 0, 2**(capacity-1).bit_length()-1
        self.max, self.sum_tree = 1, np.zeros((self.tree_start + capacity, ), dtype=np.float32)
        self.data = np.array([blank_sard]*capacity, dtype=sard_dtype)

    def total(self):
        return self.sum_tree[0]

    def get(self, data_idx):
        return self.data[data_idx % self.tree_capacity]

    def find(self, prios):
        idxs = self._retrieve(np.zeros(prios.shape, dtype=np.int32), prios)
        data_idxs = idxs - self.tree_start
        return (self.sum_tree[idxs], data_idxs, idxs)

    def append(self, sard, prio) -> None:
        self.data[self.idx] = sard
        self._update_prio(self.idx+self.tree_start, prio)
        self.idx = (self.idx+1)%self.tree_capacity # FIFO
        self.full = self.full or self.idx==0

    def _retrieve(self, idxs, prios):
        children_idxs = (idxs * 2 + np.expand_dims([1, 2], axis=1))
        if children_idxs >= self.sum_tree.shape[0]:
            return idxs
        elif children_idxs >= self.tree_start:
            children_idxs = np.minimum(children_idxs, self.sum_tree.shape[0]-1)
        left_children_prios = self.sum_tree[children_idxs[0]]
        succesor_choices = np.greater(prios, left_children_prios).astyp(p.int32)
        succesor_idxs = children_idxs[succesor_choices, np.arange(idxs.size)]
        succesor_prios = prios - succesor_choices*left_children_prios
        return self._retrieve(succesor_idxs, succesor_prios)

    def _update_prio(self, idx, prio) -> None:
        self.sum_tree[idx] = prio
        self._propagate_prio(idx)
        self.max = max(prio, self.max)

    def _update_prios(self, idxs, prios) -> None:
        self.sum_tree[idxs] = prios
        self._propagate_prios(idxs)
        self.max = max(np.max(prios), self.max)

    def _propagate_prio(self, idx) -> None:
        parent_idx = (idx-1)//2
        left_idx, right_idx = 2*parent_idx + 1, 2*parent_idx + 2
        self.sum_tree[parent_idx] = self.sum_tree[left_idx] + self.sum_tree[right_idx]
        if parent_idx != 0: self._propagate_prio

    def _propagate_prios(self, idxs) -> None:
        parent_idxs = np.unique((idxs-1)//2)
        self._update_children_prios(parent_idxs)
        if parent_idxs[0] != 0: self._propagate_prios

    def _update_children(self, idxs) -> None:
        children_idxs = idxs * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[idxs] = np.sum(self.sum_tree[children_idxs], axis=0)



class PixelPER: # Done
    def __init__(
        self,
        configs,
        hyper_para,
        seed = 0, device = 'cpu'):
        self.seed, self._device_ = seed, device
        self.capacity, self.batch_size = configs['capacity'], configs['batch-size']
        self.history, self.n_steps = hyper_para['history'], hyper_para['n-steps']
        self.gamma, self.omega, self.beta = hyper_para['gamma'], hyper_para['omega'], hyper_para['beta']
        self.t, self.transitions = 0, SegmentTree(self.capacity)
        self.gamma_n = T.tensor([self.gamma**i for i in range(self.n_steps)], dtype=T.float32, device=self._device_)

    def _size(self):
        return self.capacity if self.transitions.full else self.transitions.idx

    def append_sard(self, s, a, r, d) -> None: # Done
        self.transitions.append((self.t, s[-1], a, r, d), self.transitions.max)
        self.t = 0 if d else self.t+1

    def update_prios(self, idxs, prios) -> None: # Done
        prios = np.power(prios, self.omega)
        self.transitions._update_prios(idxs, prios)

    def sample_batch(self, batch_size) -> Dict: # Done
        total_prios = self.transitions.total()
        segment_batch = self._sample_batch_from_segments(batch_size, total_prios)
        probs = segment_batch['probs'] / total_prios
        capacity = self.capacity if self.transitions.full else self.transitions.idx
        weights = (capacity*probs) ** -self.beta
        weights_normz = T.tensor(weights/weights.max(), dtype=np.float32, device=self._device_)
        batch = dict(tree_idxs=segment_batch['tree_idxs'],
                     observations=segment_batch['observations'],
        			 actions=segment_batch['actions'],
        			 returns=segment_batch['returns'],
        			 observations_next=segment_batch['observations_next'],
        			 terminals=segment_batch['terminals'],
                     importance_ws=weights_normz)
        return batch

    def _sample_batch_from_segments(self, batch_size, total_prios): # Done
        seg_len = total_prios / batch_size
        seg_i = np.arange(batch_size) * seg_len
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, seg_len, [batch_size]) + seg_i
            probs, idxs, tree_idxs = self.transitions.find(samples)
            if np.all((self.transitions.idx - idxs) % self.capacity > self.n_steps)\
            and np.all((idxs - self.transitions.idx) % self.capacity >= self.history)\
            and np.all(probs != 0):
                valid = True
        transitions = self._get_transitions(idxs)
        observations = T.tensor(transitions['observation'][:, :self.history], dtype=T.float32, device=self._device_)#.div_(255)
        observations_next = T.tensor(transitions['observation'][:, self.n_steps:self.n_steps+self.history], dtype=np.float32, device=self._device_)#.div_(255)
        actions = T.tensor(np.copy(transitions['action'][:, self.history-1]), dtype=T.int64, device=self._device_)
        rewards = T.tensor(np.copy(transitions['reward'][:, self.history-1:-1]), dtype=T.float32, device=self._device_)
        returns = T.matmul(rewards, self.gamma_n)
        terminals = T.tensor(np.expand_dims(transitions['terminal'][:, self.history+self.n_steps-1], axis=1), dtype=T.float32, device=self._device_)
        batch = dict(probs=probs,
                     idxs=idxs,
                     tree_idxs=tree_idxs,
                     observations=observations,
        			 actions=actions,
        			 returns=returns,
        			 observations_next=observations_next,
        			 terminals=terminals)
        return batch

    def _get_transitions(self, idxs) -> None: # Done
        transitions_idxs = np.arange(-self.history+1, self.n+1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transitions_idxs)
        transitions_t0 = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_t0, dtype=bool)
        for t in range(self.history-2, -2, -1): # (t-1, t-2, t-3)-1
            blank_mask[:, t] = np.logical_or(blank_mask[:, t+1], transitions_t0[:, t+1])
        for t in range(self.history, self.history+self.n_steps): # (t+1, t+2, t+3)
            blank_mask[:, t] = np.logical_or(blank_mask[:, t-1], transitions_t0[:, t])
        transitions[blank_mask] = blank_sard

    def __next__(self): # Done
        if current_idx == self.capacity: raise StopIteration
        sard = self.transitions.data[np.arange(self.current_idx-self.history+1, self.current_idx+1)]
        transitions_t0 = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_t0, dtype=bool)
        for t in reversed(range(self.history-1)):
            blank_mask[t] = np.logical_or(blank_mask[t+1], transitions_t0[t+1])
        transitions[blank_mask] = blank_sard
        observation = T.tensor(transitions['observation'], dtype=T.float32, device=self._device_).div_(255)
        self.current_idx += 1
        return observation

    def __iter__(self): # Done
        self.current_idx = 0
        return self
