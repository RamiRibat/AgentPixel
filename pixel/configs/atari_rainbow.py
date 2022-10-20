

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
        'reward-clip': False,
        'max-steps': int(27e3), # per episode
        'max-frames': int(108e3), # per episode
        'pre-process': ['AtariPreprocessing'],
    },

    'learning': {
        'total-steps': int(50e6), # 50e6 steps X4 = 200e6 frames
        'init-steps': int(2e4),
        'expl-steps': int(0),
        'learn-freq': 4,
        'grad-steps': 1,
        'log-freq': 0,
        'render': False,
    },

    'evaluation': {
        'evaluate': True,
        'eval-freq': int(1e5),
        'episodes': 10,
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
            # 'alpha': 0.2,
            'omega': 0.5,
            'beta': 0.4,
            # 'prio-eps': 1e-6,
            'v-min': -10.0, #
            'v-max': 10.0,
            'atom-size': 51,
            'target-update-frequency': int(8e3),
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
        # 'buffer-type': 'per+nSteps',
        'buffer-type': 'pixel-per',
        'capacity': int(1e6),
        'batch-size': 32,
    }

}
