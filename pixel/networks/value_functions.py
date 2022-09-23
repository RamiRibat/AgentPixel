# from typing import tuple

import torch as T
nn, F = T.nn, T.nn.functional

from pixel.networks.dnns import Network, NoisyNetwork


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, net_configs, seed, device):
        super(QNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.qnet = Network(obs_dim, act_dim, net_configs).to(device)
        # self.to(device)
        self.optimizer = eval(optimizer)(self.parameters())

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.qnet(observation)


class NDCQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, atom_size, v_min, v_max, net_configs, seed, device):
        super(QNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.act_dim, self.atom_size = act_dim, atom_size
        self.support = support = T.linspace(v_min, v_max, atom_size)
        self.feature_layer = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        self.v_net = NoisyNetwork(obs_dim, 1*atom_size, net_configs)
        self.adv_net = NoisyNetwork(obs_dim, act_dim*atom_size, net_configs)
        self.to(device)
        self.optimizer = eval(optimizer)(self.parameters(), lr)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        distribution = self.distribution(observation)
        return T.sum(distribution*self.support, dim=2)

    def distribution(self, observation: T.Tensor) -> T.Tensor:
        feature = self.feature_layer(observation)
        value = self.v_net(feature).view(-1, 1, self.atom_size)
        advantage = self.adv_net(feature).view(-1, self.act_dim, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1).clamp(min=1e-3)




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
