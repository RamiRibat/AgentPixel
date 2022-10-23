

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
        'name': 'CartPole-v1',
        'domain': 'gym',
        'horizon': int(500),
        'state': 'discrete',
        'action': 'discrete',
        'vectorized': False,
    },

    'learning': {
        'steps': int(2e4),
        'init-steps': int(128),
        'expl-steps': int(0),
        'frequency': 1,
        'grad-steps': 1,
        'render': False,
    },

    'evaluation': {
        'evaluate': True,
        'frequency': 100,
        'episodes': 5,
        'render': False,
    },

    'algorithm': {
        'name': 'Rainbow',
        # 'model': None,
        # 'on-policy': False,
        # 'model-based': False,
        'hyper-parameters': {
            'history': 1,
            'gamma': 0.99,
            'alpha': 0.2,
            'beta': 0.6,
            'prio-eps': 1e-6,
            'v-min': 0.0, #
            'v-max': 200.0,
            'atom-size': 51,
            'n-steps': 3,
            'target-update-frequency': 100,
        }
    },

    'actor': {
        'type': 'greedy'
    },

    'critic': {
        'type': 'Q-functions',
        'network': {
            'optimizer': 'Adam',
            'op_activation': 'Identity',
            'activation': 'ReLU',
            'arch': [128, 128],
            'lr': 1e-3,
        }
    },

    'data': {
        # 'buffer_type': 'per+nSteps',
        'obs-type': 'numerical',
        # 'buffer-type': 'nStepsPER',
        'buffer-type': 'PER',
        'buffer-size': int(1e4),
        'capacity': int(1e4),
        'batch-size': 128,

    }

}
