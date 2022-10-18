# from typing import tuple

import torch as T
nn, F = T.nn, T.nn.functional

from pixel.networks.dnns import Network, NoisyNetwork


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, net_configs, seed, device):
        print('Initialize QNetwork')
        super(QNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.qnet = Network(obs_dim, act_dim, net_configs)
        self.to(device)
        self.optimizer = eval(optimizer)(self.parameters(), lr)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        # print('q-observation: ', observation.shape)
        q = self.qnet(observation)
        # print('q-value: ', q.shape)
        return q


class NDCQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, atom_size, v_min, v_max, net_configs, seed, device):
        print('Initialize NDCQNetwork')
        super(NDCQNetwork, self).__init__()
        optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
        self.act_dim, self.atom_size = act_dim, atom_size
        self.support = support = T.linspace(v_min, v_max, atom_size)
        self.feature_layer = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        self.v_net = NoisyNetwork(128, 1*atom_size, net_configs)
        self.adv_net = NoisyNetwork(128, act_dim*atom_size, net_configs)
        self.to(device)
        self.optimizer = eval(optimizer)(self.parameters(), lr)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        distribution = self.distribution(observation)
        q = T.sum(distribution*self.support, dim=2)
        return q

    def distribution(self, observation: T.Tensor) -> T.Tensor:
        feature = self.feature_layer(observation)
        value = self.v_net(feature).view(-1, 1, self.atom_size)
        advantage = self.adv_net(feature).view(-1, self.act_dim, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1).clamp(min=1e-3)

    def reset_noise(self):
        self.v_net.reset_noise()
        self.adv_net.reset_noise()

    def _evaluation_mode(self, mode=False):
        self.v_net._evaluation_mode(mode)
        self.adv_net._evaluation_mode(mode)




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
