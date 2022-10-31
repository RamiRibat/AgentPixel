import math
from typing import Dict, List

import torch
import torch as T # torch==1.12.1 --> torch==1.10.2
nn, F = T.nn, T.nn.functional


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init: float = 0.1):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(T.empty(out_features, in_features)) # reveresed
        self.weight_sigma = nn.Parameter(T.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', T.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(T.empty(out_features))
        self.bias_sigma = nn.Parameter(T.empty(out_features))
        self.register_buffer('bias_epsilon', T.empty(out_features))
        self.reset_parametrs()
        self.reset_noise()
        self.evaluation_mode = False

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
        x = T.randn(size)
        return x.sign().mul_(x.abs().sqrt_())



# Factorised NoisyLinear layer with bias
class NoisyLinear1(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

    self.evaluation_mode = False

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    # print(f'reset_noise')
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  # def forward(self, input):
  #   if self.training:
  #     return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
  #   else:
  #     return F.linear(input, self.weight_mu, self.bias_mu)



  def forward(self, x: T.Tensor) -> T.Tensor:
      if self.evaluation_mode:
          return F.linear(x,
                          self.weight_mu,
                          self.bias_mu)
      else:
          return F.linear(x,
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu   + self.bias_sigma   * self.bias_epsilon)





class NoisyNetwork(nn.Module):
    """
    Reference: Noisy Networks for Exploration (DeepMind; ICLR 2018)
    """
    def __init__(self, in_dim: int, out_dim: int, configs: Dict):
        super(NoisyNetwork, self).__init__()
        self.net = nn.Sequential(
            NoisyLinear(in_features=in_dim,
                        out_features=configs['arch'][1],
                        std_init=configs['std']),
            nn.ReLU(),
            NoisyLinear(in_features=configs['arch'][1],
                        out_features=out_dim,
                        std_init=configs['std'])
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)

    def reset_noise(self):
        for m in self.net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _evaluation_mode(self, mode=False):
        for m in self.net.modules():
            if isinstance(m, NoisyLinear):
                m.evaluation_mode = mode



# Networks w/ Visual Inputs
class Encoder(nn.Module):
    def __init__(self, in_dim: int, configs: Dict):
        super(Encoder, self).__init__()
        if configs['pre-train']:
            raise "Add a pre-trained model"
        else:
            if configs['arch'][0] == 'canonical':
                self.net = nn.Sequential(
                    nn.Conv2d(in_dim, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
                    nn.Conv2d(32,     64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
                    nn.Conv2d(64,     64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
                )
            elif configs['arch'][0] == 'data-efficient':
                self.net = nn.Sequential(
                    nn.Conv2d(in_dim, 32, kernel_size=5, stride=5, padding=0), nn.ReLU(),
                    nn.Conv2d(32,     64, kernel_size=5, stride=5, padding=0), nn.ReLU(),
                )

            self.feature_dim = configs['arch'][1]

    def forward(self, x: T.Tensor) -> T.Tensor:
        latent_features = self.net(x)
        return latent_features.view(-1, self.feature_dim)
