# from typing import tuple

import torch as T
nn, F = T.nn, T.nn.functional

from pixel.networks.dnns import Network, NoisyNetwork, Encoder


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
    def __init__(
        self,
        obs_dim, act_dim,
        configs, hyper_para,
        seed = 0, device='cpu'):
        # print('Initialize NDCQNetwork')
        super(NDCQNetwork, self).__init__()

        optimizer = 'T.optim.' + configs['optimizer']['type']
        lr = configs['optimizer']['lr']
        eps = configs['optimizer']['eps']
        # norm_clip = configs['optimizer']['norm-clip']

        self.act_dim, self.atom_size = act_dim, hyper_para['atom-size']
        self.support = T.linspace(
                            hyper_para['v-min'],
                            hyper_para['v-max'],
                            hyper_para['atom-size']).to(device)

        if obs_dim == 'pixel':
            self.feature_layer = Encoder(hyper_para['history'], configs['encoder'])
            self.feature_dim = self.feature_layer.feature_dim
        else:
            self.feature_layer = nn.Sequential(nn.Linear(obs_dim, configs['mlp']['arch'][0]), nn.ReLU())
            self.feature_dim = configs['mlp']['arch'][0]
        self.v_net = NoisyNetwork(self.feature_dim, 1*self.atom_size, configs['mlp'])
        self.adv_net = NoisyNetwork(self.feature_dim, self.act_dim*self.atom_size, configs['mlp'])
        # print('NDCQNetwork: ', self)
        self.to(device)

        self.optimizer = eval(optimizer)(self.parameters(), lr=lr, eps=eps)

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



#
# class SoftQNetworks(nn.Module):
#     def __init__(self, obs_dim, act_dim, net_configs, seed, device):
#         super(QNetwork, self).__init__()
#         optimizer, lr = 'T.optim.' + net_configs['optimizer'], net_configs['lr']
#         self.q1 = Network(obs_dim, act_dim, net_configs)
#         self.q2 = Network(obs_dim, act_dim, net_configs)
#         self.QNets = [self.q1, self.q2]
#         self.to(device)
#         self.optimizer = eval(optimizer)(self.parameters(), lr)
#
#     def forward(self, observation):
#         return tuple(Q(observation) for Q in self.QNets)
