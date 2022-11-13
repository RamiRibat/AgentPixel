

configurations = {

    'comments': {
        'Rami Ahmed*': None,
    },

    'experiment': {
        'id': None,
        'device': 'cpu',
        'print_logs': True,
        'logger': 'WB',
    },

    'environment': {
        'name': 'ALE/Alien-v5',
        'domain': 'atari',
        'state': 'pixel',
        'action': 'discrete',
        'n-envs': [2],
        'asynchronous': True,
        'reward-clip': 1,
        'n-stacks': 4,
        'frameskip': 1,
        'repeat-action-probability': 0,
        'max-steps': int(27e3), # per episode
        'max-frames': int(108e3), # per episode

        'pre-processing': {
            'noop_max': 30,
            'frame_skip': 4,
            'screen_size': 84,
            'terminal_on_life_loss': True, # training only
            'grayscale_obs': True,
            'grayscale_newaxis': False,
            'scale_obs': True, # default=False
        },
    },

    'learning': {
        'total-steps': int(200e3), # 100k in van Hasselt et al. (2019)
        'init-steps': int(2000), # 1600 in van Hasselt et al. (2019)
        'expl-steps': int(1000), # 0 in van Hasselt et al. (2019)
        'learn-freq': 1, # iteration
        'grad-steps': 1, # v1
        # 'grad-steps': 2, # v2
        'render': False,
    },

    'evaluation': {
        'evaluate': True,
        'eval-freq': int(5e3), # iteration X n-envs
        'episodes': 10,
        'render': False,
    },

    'algorithm': {
        'name': 'DERainbow',
        # 'model': None,
        # 'on-policy': False,
        # 'model-based': False,
        'hyperparameters': {
            'history': 4,
            'n-steps': 20,
            'gamma': 0.99,
            'omega': 0.5, # prio-exponent
            'beta': 0.4, # prio-weight --> 1
            'prio-eps': 1e-6,
            'v-min': -10.0, #
            'v-max': 10.0,
            'atom-size': 51,
            'target-update-frequency': int(2000), # v1
            # 'target-update-frequency': int(1000), # v2
        }
    },

    'actor': {
        'type': 'gready-noisy-nets'
    },

    'critic': {
        'type': 'Duelling Value-function',
        'network': {
            'encoder': {
                'pre-train': False,
                # 'arch': ['canonical', 3136],
                'arch': ['data-efficient', 576],
                'activation': 'ReLU',
            },
            'mlp': {
                # 'type': 'Linear',
                'type': 'NoisyLinear',
                # 'arch': [512, 512],
                'arch': [256, 256],
                'std': 0.1,
                'activation': 'ReLU',
                'op_activation': 'Identity',
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-4,
                'eps': 1.5e-4,
                'norm-clip': 10,
            },
        }
    },

    'data': {
        'obs-type': 'pixel',
        'buffer-type': 'PER',
        'capacity': int(2e5),
        'batch-size': 32, # v1/2
    }

}
