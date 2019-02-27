env = {'name': 'MountainCarContinuous-v0',
       'render': False,
       'solved_th': 80,
       'action_size': 10,
       'state_size': 6}

paths = {'logs': '../logs/mountain_car',
         'model': '../models/mountain_car/mountain_car-model'}

algo = {'max_episodes': 5000,
        'max_steps': 10000000,
        'discount_factor': 0.99}

network = {'policy_learning_rate': 0.001,
           'value_net_learning_rate':  0.001,
           'learning_rate_decay': 0.99,
           'max_action_size': 10,
           'max_state_size': 6}
