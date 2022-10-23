

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
        'name': 'ALE/Pong-v5',
        'domain': 'atari',
        'state': 'pixel',
        'action': 'discrete',
        'n-envs': [0],
        'asynchronous': True,
        'n-stacks': 4,
        'frame-skip': 4,
        'reward-clip': 1,
        'max-steps': int(27e3), # per episode
        'max-frames': int(108e3), # per episode
        'pre-process': ['AtariPreprocessing'],
    },

    'learning': {
        # 'total-steps': int(50e2), # 50e6 steps X4 = 200e6 frames
        # 'init-steps': int(20e1),
        'total-steps': int(50e6), # 50e6 steps X4 = 200e6 frames
        'init-steps': int(20e3),
        'expl-steps': int(10e3),
        # 'learn-freq': 4, # iteration
        'learn-freq': 1, # iteration
        'grad-steps': 1,
        'log-freq': 0,
        'render': False,
    },

    'evaluation': {
        'evaluate': True,
        # 'eval-freq': int(1e5),
        # 'episodes': 10,
        # 'eval-freq': int(1e5),
        'eval-freq': int(1e3), # iteration X n-envs
        'episodes': 5,
        'render': False,
    },

    'algorithm': {
        'name': 'Rainbow',
        # 'model': None,
        # 'on-policy': False,
        # 'model-based': False,
        'hyper-parameters': {
            'history': 4,
            'n-steps': 3,
            'gamma': 0.99,
            'omega': 0.5, # prio-exponent
            'beta': 0.4, # prio-weight
            'prio-eps': 1e-6,
            'v-min': -10.0, #
            'v-max': 10.0,
            'atom-size': 51,
            'target-update-frequency': int(200),
            # 'target-update-frequency': int(8e3),
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
                'arch': ['canonical', 3136],
                # 'arch': ['data-efficient', 576],
                'activation': 'ReLU',
            },
            'mlp': {
                # 'type': 'Linear',
                'type': 'NoisyLinear',
                'arch': [512, 512],
                'std': 0.1,
                'activation': 'ReLU',
                'op_activation': 'Identity',
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 65e-5,
                'eps': 1.5e-4,
                'norm-clip': 10,
            },
        }
    },

    'data': {
        'obs-type': 'pixel',
        'buffer-type': 'PER',
        'capacity': int(1e6),
        'batch-size': 32,
        'batch-size': 128,
    }

}
