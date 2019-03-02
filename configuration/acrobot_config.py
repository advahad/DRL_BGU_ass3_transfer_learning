env = {'name': 'Acrobot-v1',
       'render': False,
       'solved_th': -100, #TODO: cahnge to 100
       'action_size': 3,
       'max_episode_steps': 500}

paths = {'logs': '../logs/acrobot',
         'model': '../models/acrobot/acrobot-model',
         'checkpoint': '../models/acrobot'}

algo = {'max_episodes': 5000,
        'max_steps': 200,
        'discount_factor': 0.99}

network = {'policy_learning_rate': 0.0001,
           'value_net_learning_rate':  0.001,
           'learning_rate_decay': 0.995,
           'max_action_size': 10,
           'max_state_size': 6}