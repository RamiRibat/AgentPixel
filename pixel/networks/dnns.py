import math
from typing import Dict

import torch as T
nn, F = T.nn, T.nn.functional


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(T.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(T.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', T.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(T.empty(out_features))
        self.bias_sigma = nn.Parameter(T.empty(out_features))
        self.register_buffer('bias_epsilon', T.empty(out_features))
        self.reset_parametrs()
        self.reset_noise()

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.evaluation_mode:
            return F.linear(x,
                            self.weight_mu,
                            self.bias_mu)
        else:
            return F.linear(x,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu   + self.bias_sigma   * self.bias_epsilon)

    def reset_parametrs(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> T.Tensor:
        x = T.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())



class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, configs: Dict):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        # self.optimizer = T.optim.Adam(self.net.parameters(), lr=0.0001)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class NoisyNetwork(nn.Module):
    """
    Reference: Noisy Networks for Exploration (DeepMind; ICLR 2018)
    """
    def __init__(self, in_dim: int, out_dim: int, net_configs: Dict):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.NoisyLinear(in_dim, 128),
            nn.ReLU(),
            nn.NoisyLinear(128, 128),
            nn.ReLU(),
            nn.NoisyLinear(128, out_dim)
        )
        # self.optimizer = T.optim.Adam(self.net.parameters(), lr=0.0001)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)



# Networks w/ Visual Inputs
class Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Encoder, self).__init__()
        pass

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class VisualNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Encoder, self).__init__()
        pass

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)
