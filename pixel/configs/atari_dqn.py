

configurations = {

    'comments': {
        'Rami': None,
    },

    'experiment': {
        'id': None,
        'device': 'cpu',
        'print_logs': True,
        'logger': 'WB',
    },

    'environment': {
        # 'name': 'ALE/Asterix-v5',
        # 'name': 'ALE/Boxing-v5',
        # 'name': 'ALE/Breakout-v5',
        'name': 'ALE/Pong-v5',
        'domain': 'atari',
        'state': 'pixel',
        'action': 'discrete',
        'n-envs': 0,
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
        'init-steps': int(1e3),
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
        'name': 'DQN',
        # 'model': None,
        # 'on-policy': False,
        # 'model-based': False,
        'hyper-parameters': {
            'history': 4,
            'n-steps': 1,
            'gamma': 0.99,
            'init-epsilon': 1.0,
            'max-epsilon': 1.0,
            'min-epsilon': 0.1,
            'epsilon-decay': 1/1000,
            'target-update-frequency': 200,
        }
    },

    'actor': {
        'type': 'eps-greedy'
    },

    'critic': {
        'type': 'Q-function',
        'network': {
            'encoder': {
                'pte-train': False,
                'arch': ['canonical', 3136],
                # 'arch': ['data-efficient', 576],
                'activation': 'ReLU',
            },
            'mlp': {
                'type': 'Linear',
                'arch': [512, 512],
                'std': 0.1,
                'activation': 'ReLU',
                'op_activation': 'Identity',
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 65e-5,
                'eps': 1.5e-4,
                'norm-clip': 5,
            },
        }
    },

    'data': {
        'buffer-type': 'simple',
        'buffer-size': int(1e6),
        'batch-size': 32,
    }

}
