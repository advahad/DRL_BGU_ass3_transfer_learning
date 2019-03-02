env = {'name': 'CartPole-v1',
       'render': False,
       'solved_th': 475,
       'action_size': 2,
       'state_size': 4,
       'max_episode_steps': None}

paths = {'logs': '../logs/cartpole',
         'model': '../models/cartpole/cartpole-model'}

algo = {'max_episodes': 5000,
        'max_steps': 10000000,
        'discount_factor': 0.99}

network = {'policy_learning_rate': 0.001,
           'value_net_learning_rate':  0.001,
           'learning_rate_decay': 0.999,
           'max_action_size': 10,
           'max_state_size': 6}

