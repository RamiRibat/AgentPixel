

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
        'name': 'CartPole-v0',
        'domain': 'gym',
        'horizon': int(500),
        'state': 'discrete',
        'action': 'discrete',
        'vectorized': False,
    },

    'learning': {
        'steps': int(2e4),
        # 'epoch_steps': int(1e3),
        'init_steps': int(128),
        'expl_steps': int(0),
        'frequency': 1,
        'grad_steps': 1,
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
            'gamma': 0.99,
            'alpha': 0.2,
            'beta': 0.6,
            'prior-epsilon': 1e-6,
            'v-min': 0.0,
            'v-max': 200.0,
            'atom-size': 51,
            'n-steps': 3,
            'target_update_frequency': 100,
        }
    },

    'actor': {
        'type': 'greedy'
    },

    'critic': {
        'type': 'Douple Q-functions',
        'network': {
            'optimizer': 'Adam',
            'op_activation': 'Identity',
            'activation': 'ReLU',
            'arch': [128, 128],
            'lr': 1e-3,
        }
    },

    'data': {
        'buffer_type': 'simple-numpy',
        'buffer_size': int(1e4),
        'batch_size': 128,

    }

}
