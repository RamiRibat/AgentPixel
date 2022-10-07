import psutil
import argparse
import time, datetime
from tqdm import tqdm, trange

import numpy as np
import gym
import wandb


# if __name__ == '__main__':
#     # env = gym.make("ALE/Pong-v5")
#     # observation, info = env.reset()
#     # print('init-observation: ', observation)
#     # print('init-info: ', info)
#     # print('observation_space: ', env.observation_space)
#     # print('action_space: ', env.action_space)
#
#     envs = gym.vector.make("ALE/Pong-v5", num_envs=3)
#     observation, info = envs.reset()
#     print('init-observation: ', observation.shape)
#     print('init-info: ', info)
#     print('observation_space: ', envs.observation_space)
#     print('action_space: ', envs.action_space)

# if __name__ == '__main__':
#     envs = gym.vector.make("ALE/Pong-v5", num_envs=20)
#     o, info = envs.reset()
#     LT = trange(1, 100000+1, desc="ALE/Pong-v5", position=0)
#     CPU = tqdm(total=100, desc='CPU %', position=1, colour='RED')
#     RAM = tqdm(total=100, desc='RAM %', position=2, colour='BLUE')
#     with CPU, RAM:
#         for t in LT:
#             a = envs.action_space.sample()
#             o_next, r, terminated, truncated, info = envs.step(a)
#             # print(f'[{t}] | terminated:{terminated} | truncated:{truncated}')
#             if t%1000==0:
#                 CPU.n = psutil.cpu_percent()
#                 CPU.refresh()
#                 RAM.n = psutil.virtual_memory().percent
#                 RAM.refresh()
#     envs.close()



def main(env, num, seed, device, wb):

    alg_name = 'NL' # configurations['algorithm']['name']
    env_name = env #configurations['environment']['name']
    num_envs = num
    env_domain = 'Atari' # configurations['environment']['domain']

    group_name = f"Mac-AP-{env_domain}-{env_name}-X{num_envs}"
    # group_name = f"V-AP-{env_domain}-{env_name}-X{num_envs}"
    # group_name = f"Q-AP-{env_domain}-{env_name}-X{num_envs}"
    exp_prefix = f"{group_name}-{alg_name}-seed:{seed}"

    print('=' * 50)
    print(f'Start of an RL experiment')
    print(f"\t Environment: {env_name} X {num}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            project=f'RL-WC'
        )

    LS = int(1e5)

    # envs = gym.vector.make(env_name, num_envs=num)
    if num == 0:
        num = 1
        envs = gym.make("ALE/Pong-v5")
        # envs = gym.make("Pong-v4")
    else:
        envs = gym.vector.make(env_name, num_envs=num)

    o, info = envs.reset()
    env_steps = 0
    LT = trange(1, LS+1, desc=env_name, position=0)
    # SPS = tqdm(desc='SPS', position=1, colour='PURPLE')
    CPU = tqdm(total=100, desc='CPU %', position=1, colour='RED')
    RAM = tqdm(total=100, desc='RAM %', position=2, colour='BLUE')
    CPUList, RAMList, SPSList = [], [], []
    logs = dict()

    with CPU, RAM:
        start_time_real = time.time()
        for t in LT:
            a = envs.action_space.sample()
            o_next, r, terminated, truncated, info = envs.step(a)
            env_steps += num
            if t%500==0:
                cur_time_real = time.time()
                total_time_real = cur_time_real - start_time_real
                sps = env_steps//total_time_real
                # SPS.n = sps
                # SPS.refresh()
                SPSList.append(sps)
                # print(f'Time={total_time_real} | env_steps={env_steps} | sps={sps}')

                CPU.n = psutil.cpu_percent()
                CPU.refresh()
                RAM.n = psutil.virtual_memory().percent
                RAM.refresh()
                CPUList.append(CPU.n)

                logs['hardware/cpu                        '] = CPU.n
                logs['hardware/cpu-avg                    '] = np.mean(CPUList)
                logs['hardware/ram                        '] = RAM.n
                logs['time/total                          '] = total_time_real
                logs['time/sps                            '] = sps
                logs['time/sps-avg                        '] = np.mean(SPSList)
                LT.set_postfix({'SPS': sps})
                if wb: wandb.log(logs, step=env_steps)
            if env_steps >= LS: break
    envs.close()


    print('\n')
    print('End of the RL experiment')
    print('=' * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-env', type=str, default="ALE/Pong-v5")
    parser.add_argument('-num', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-wb', action='store_true')

    args = parser.parse_args()

    env = args.env
    num = args.num
    seed = args.seed
    device = args.device
    wb = args.wb

    main(env, num, seed, device, wb)
