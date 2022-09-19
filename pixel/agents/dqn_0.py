from typing import Any
from random import sample
from dataclasses import dataclass
from tqdm import tqdm, trange
import time

import gym

import torch as T
nn, F = T.nn, T.nn.functional

import wandb



@dataclass
class SARST:
    state: Any
    action: int
    reward: float
    state_next: Any
    terminal: bool

# class ReplayBuffer:
#     def __init__(self, buffer_size=int(1e4)):
#         self.buffer_size = buffer_size
#         self.buffer = []
#         pass
#
#     def insert(self, sarst):
#         self.buffer.append(sarst)
#         self.buffer = self.buffer[-self.buffer_size:]
#         pass
#
#     def sample(self, n_samples):
#         assert n_samples <= len(self.buffer)
#         return sample(self.buffer, n_samples)
#         pass

class ReplayBuffer:
    "Simple Replay Buffer (Numpy)"
    def __init__(self, obs_dim: int, act_dim: int, max_size: int, batch_size: int = 32 seed=0, device='cpu'):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.obs_next_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.ter_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size, self.batch_size = 0, 0, max_size, batch_size

    def store_sarsd(
            self,
            o: np.ndarray,
            a: np.ndarray,
            r: float,
            o_next: np.ndarray,
            d: bool):
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs_next_buf[self.ptr] = o_next
        self.ter_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        return batch

    def __len__(self) -> int:
        return self.size


class DeepNetwork(nn.Module):
    def __init__(self, ip_dim: int, op_dim: int):
        super(DeepNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ip_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, op_dim)
        )
        self.optimizer = T.optim.Adam(self.net.parameters(), lr=0.0001)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, ):
        pass
    pass


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())

def train_model(sarst_batch, model, model_target):
    observations = T.stack(([T.as_tensor(sarst.state) for sarst in sarst_batch]))
    actions = [sarst.action for sarst in sarst_batch]
    rewards = T.stack(([T.as_tensor([sarst.reward]) for sarst in sarst_batch]))
    observations_next = T.stack(([T.as_tensor(sarst.state_next) for sarst in sarst_batch]))
    terminals = T.stack(([T.as_tensor([sarst.terminal]) for sarst in sarst_batch]))
    # print('term: ', terminals)
    # print(f'dimensions\nobs={observations.shape} | act={len(actions)} | rew={rewards.shape} | ter={terminals.shape}')

    oh_actions = F.one_hot(T.LongTensor(actions), n_actions)

    with T.no_grad(): q_target_next = model_target(observations_next).max(-1)[0]
    q_target = rewards.squeeze() + (~terminals.squeeze())*q_target_next
    q = model(observations)
    # print(f'qt_next={q_target_next.shape} | qt={q_target.shape} | q={T.sum(q*oh_actions, -1).shape}')

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(q_target, T.sum(q*oh_actions, -1))
    # loss = (q_target - T.sum(q*oh_actions, -1)).mean()

    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    return loss





# def main(exp_prefix, config, seed, device, wb):
#     print('Start a DQN experiment...\n')

def main(exp_prefix, config, seed, device, wb):

    print('Start a DQN experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_domain = configs['environment']['domain']

    group_name = f"{env_name}-{alg_name}-a" # H < -2.7
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project=f'vector',
            config=configs
        )

    agent = DQN(exp_prefix, configs, seed, device, wb)

    # agent.learn()

    print('\n')
    print('... End the PPO experiment')


# WandB = False
# WandB = True

if __nameaa__ == '__mainaa__':
    if WandB: wandb.init(project='vector', name='CarPole-v1-DQN-3')
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    observation, info = env.reset(seed=42)
    rb = ReplayBuffer()
    dqn_model = DQNModel(env.observation_space.shape, env.action_space.n)
    dqn_model_target = DQNModel(env.observation_space.shape, env.action_space.n)

    update_target_model(dqn_model, dqn_model_target)

    total_steps, training_frequency, target_update_frequency = 0, 100, 100
    min_rb_size, sample_size = 10000, 2500
    T_Max = int(1e5)
    gamma, eps_decay = 0.99, 1/2000

    # tq = tqdm()

    try:
        # while True:
        for t in trange(1, T_Max + 1):
            # tq.update(1)
            action = env.action_space.sample()
            observation_next, reward, terminal, truncated, info = env.step(action=action)
            # print('terminal: ', terminal)
            rb.insert(SARST(observation, action, reward, observation_next, terminal))
            observation = observation_next
            if terminal or truncated: observation, info = env.reset()
            total_steps += 1
            if (len(rb.buffer) >= min_rb_size) and (total_steps % training_frequency == 0):
                loss = train_model(rb.sample(sample_size), dqn_model, dqn_model_target)
                # print(f'Steps: {total_steps} | Loss: {loss.detach().item()}')
                if WandB: wandb.log({'loss': loss.detach().item()}, step=total_steps)
                if total_steps % target_frequency == 0: update_target_model(dqn_model, dqn_model_target)
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-exp_prefix', type=str)
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=str)
    parser.add_argument('-device', type=str)
    parser.add_argument('-wb', type=str)

    args = parser.parse_args()

    exp_prefix = args.exp_prefix
    sys.path.append(f"{os.getcwd()}/configs")
    config = importlib.import_module(args.cfg)
    seed = int(args.seed)
    device = args.device
    wb = eval(args.wb)

    main(exp_prefix, config, seed, device, wb)
