import time

import gym
import numpy as np
import tensorflow as tf

from networks.critic_network import StateValueNetwork
from networks.policy_network import PolicyNetwork
from networks.actor_critic_training import train_models, NetworkParams, AlgorithmParams

# initializations
start = time.time()
env = gym.make('Acrobot-v1')
np.random.seed(1)

# define paths
LOGS_PATH = './logs/actor-critic/Acrobot-v1'
MODEL_PATH = './models/acrobot/Acrobot-v1-model'

# define hyper parameters
action_size = env.action_space.n
max_action_size = 10
max_state_size = 6

max_episodes = 5000
max_steps = 500
discount_factor = 0.99
policy_learning_rate = 0.0001
value_net_learning_rate = 0.001
learning_rate_decay = 0.995
solved_th = -100

render = False

# networks initializations
tf.reset_default_graph()
policy = PolicyNetwork(max_state_size, max_action_size)
state_value_network = StateValueNetwork(max_state_size, 1, value_net_learning_rate)

# params initializations
network_params = NetworkParams(max_state_size, action_size, max_action_size)
algo_params = AlgorithmParams(env, render, max_episodes, max_steps, discount_factor, policy_learning_rate,
                              learning_rate_decay, solved_th)

# start trainig
train_models(policy, state_value_network, network_params, algo_params, LOGS_PATH, MODEL_PATH)

# calc time
end = time.time()
total_time = end - start
print("total running time %f" % total_time)
