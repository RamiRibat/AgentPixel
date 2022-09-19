# from typing import tuple

import torch as T
nn, F = T.nn, T.nn.functional

from pixel.networks.dnns import Network, NoisyNetwork


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, net_configs, seed, device):
        super(QNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.qnet = Network(obs_dim, act_dim, net_configs)
        self.to(device)
        self.optimizer = eval(optimizer)(self.parameters(), lr)

    def forward(self, observation):
        return self.qnet(observation)


class SoftQNetworks(nn.Module):
    def __init__(self, obs_dim, act_dim, net_configs, seed, device):
        super(QNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.q1 = Network(obs_dim, act_dim, net_configs)
        self.q2 = Network(obs_dim, act_dim, net_configs)
        self.QNets = [self.q1, self.q2]
        self.to(device)
        self.optimizer = eval(optimizer)(self.parameters(), lr)

    def forward(self, observation):
        return tuple(Q(observation) for Q in self.QNets)
