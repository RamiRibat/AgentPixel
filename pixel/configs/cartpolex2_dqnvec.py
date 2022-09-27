

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
        'n-envs': 2,
        # 'n-envs': 4,
        # 'n-envs': 8,
        # 'n-envs': 16,
    },

    'learning': {
        'steps': int(5e4),
        # 'epoch_steps': int(1e3),
        'init_steps': int(32),
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
        'name': 'DQNVEC',
        # 'model': None,
        # 'on-policy': False,
        # 'model-based': False,
        'hyper-parameters': {
            'n_steps': 1,
            'gamma': 0.99,
            'init-epsilon': 1.0,
            'max-epsilon': 1.0,
            'min-epsilon': 0.1,
            'epsilon-decay': 1/2000,
            'target_update_frequency': 100,
        }
    },

    'actor': {
        'type': 'eps-greedy'
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
        'buffer_type': 'simple',
        'buffer_size': int(1e3),
        'batch_size': 32,

    }

}
