# from typing import tuple

import torch as T
nn, F = T.nn, T.nn.functional

from pixel.networks.dnns import NoisyNetwork, Encoder
from pixel.networks.dnns import NoisyLinear




class NDCQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim, act_dim,
        configs, hyper_para,
        seed = 0, device='cpu'):
        print('Initialize NDCQNetwork')
        super(NDCQNetwork, self).__init__()

        optimizer = 'T.optim.' + configs['optimizer']['type']
        lr = configs['optimizer']['lr']
        eps = configs['optimizer']['eps']

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

        self.v_net   = NoisyNetwork(self.feature_dim, 1*self.atom_size,            configs['mlp'])
        self.adv_net = NoisyNetwork(self.feature_dim, self.act_dim*self.atom_size, configs['mlp'])
        # mlp = configs['mlp']
        # self.fc_v_h = NoisyLinear(in_features=self.feature_dim, out_features=mlp['arch'][1], std_init=mlp['std'])
        # self.fc_v_z = NoisyLinear(in_features=mlp['arch'][1], out_features=1*self.atom_size, std_init=mlp['std'])
        # self.fc_a_h = NoisyLinear(in_features=self.feature_dim, out_features=mlp['arch'][1], std_init=mlp['std'])
        # self.fc_a_z = NoisyLinear(in_features=mlp['arch'][1], out_features=self.act_dim*self.atom_size, std_init=mlp['std'])

        self.to(device)

        self.optimizer = eval(optimizer)(self.parameters(), lr=lr, eps=eps)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        q_probs = self.q_probs(observation)
        q_values = (q_probs * self.support).sum(2)
        q_actions = q_values.argmax(1)
        # print(f'QN.forward: q_probs={q_probs.shape}') # [N, A, Z]
        # print(f'QN.forward: q_values={q_values.shape}') # [N, A]
        # print(f'QN.forward: q_actions={q_actions.shape}') # [N]
        return q_values, q_actions

    def q_probs(self, observation: T.Tensor, action: T.Tensor = None, log = False) -> T.Tensor:
        q_atoms = self.distribution(observation)
        # print(f'QN.q_probs: observation={observation.shape}') # [N, Stacks, H, W]
        # print(f'QN.q_probs: q_atoms={q_atoms.shape}') # [N, A, Z]
        if log:
            log_probs = F.log_softmax(q_atoms, dim=2)
            if action is None:
                return log_probs
            else:
                return log_probs[range(action.shape[0]), action]
        else:
            probs = F.softmax(q_atoms, dim=2)
            if action is None:
                return probs
            else:
                return probs[range(action.shape[0]), action]

    def distribution(self, observation: T.Tensor) -> T.Tensor:
        # observation = observation / 255.
        feature = self.feature_layer(observation)
        value = self.v_net(feature).view(-1, 1, self.atom_size)
        advantage = self.adv_net(feature).view(-1, self.act_dim, self.atom_size)
        # value = self.fc_v_z(F.relu(self.fc_v_h(feature))).view(-1, 1, self.atom_size)
        # advantage = self.fc_a_z(F.relu(self.fc_a_h(feature))).view(-1, self.act_dim, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_atoms

    def reset_noise(self):
        # self.v_net.reset_noise()
        # self.adv_net.reset_noise()
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    # def _evaluation_mode(self, mode=False):
    #     # self.v_net._evaluation_mode(mode)
    #     # self.adv_net._evaluation_mode(mode)
    #     for name, module in self.named_children():
    #         if 'fc' in name:
    #             module.reset_noise()
