import tensorflow as tf
from networks.actor_critic_training import train_mountain_car_model
from networks.policy_network import PolicyNetwork
from networks.critic_network import StateValueNetwork
from networks.actor_critic_training import AlgorithmParams, NetworkParams
import numpy as np
import gym



# env setup
env = gym.make('mountain_car')
env._max_episode_steps = 7000
env.seed(1)
np.random.seed(1)

# CONFIGURATIONS
V_NET_LAYER_SIZE = 20
POLICY_NET_LAYER_SIZE = 20  # may be changed to 12

LOGS_PATH = './logs/actorcritic/transfer/mountain_car'

# Define hyper parameters
state_size = 2
action_size = 10
# action_size = env.action_space.n
max_state_size = 6

max_episodes = 5000
max_steps = 10000000
discount_factor = 0.99
policy_learning_rate = 0.001
value_net_learning_rate = 0.001
learning_rate_decay = 0.99  # may be 0.999
solved_th = 80

bin_values = [-0.7, -0.6, -0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7]

render = False

algo_params = AlgorithmParams(env, render, max_episodes, max_steps, discount_factor, policy_learning_rate,
                              learning_rate_decay, solved_th)

net_params = NetworkParams(max_state_size, action_size, action_size)

if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("../models/cartpole/cartpole-model.meta")
        saver.restore(sess, tf.train.latest_checkpoint('../models/cartpole'))
        graph = tf.get_default_graph()

        policy = PolicyNetwork(max_state_size, action_size, name="policy_network_new")
        state_value_network = StateValueNetwork(max_state_size, 1, value_net_learning_rate, name='state_value_net_new')

        # copy weights from trained model to new models
        # policy
        p_W1 = graph.get_tensor_by_name("policy_network/W1:0")
        p_b1 = graph.get_tensor_by_name("policy_network/b1:0")

        p_W2 = graph.get_tensor_by_name("policy_network/W2:0")
        p_b2 = graph.get_tensor_by_name("policy_network/b2:0")

        p_W3 = graph.get_tensor_by_name("policy_network/W3:0")
        p_b3 = graph.get_tensor_by_name("policy_network/b3:0")

        sess.run([tf.assign(policy.W1, p_W1), tf.assign(policy.b1, p_b1),
                  tf.assign(policy.W2, p_W2), tf.assign(policy.b2, p_b2),
                  tf.assign(policy.W3, p_W3), tf.assign(policy.b3, p_b3)])


        # value network
        p_W1 = graph.get_tensor_by_name("state_value_network/W1:0")
        p_b1 = graph.get_tensor_by_name("state_value_network/b1:0")

        p_W2 = graph.get_tensor_by_name("state_value_network/W2:0")
        p_b2 = graph.get_tensor_by_name("state_value_network/b2:0")

        p_W3 = graph.get_tensor_by_name("state_value_network/W3:0")
        p_b3 = graph.get_tensor_by_name("state_value_network/b3:0")

        sess.run([tf.assign(state_value_network.W1, p_W1), tf.assign(state_value_network.b1, p_b1),
                  tf.assign(state_value_network.W2, p_W2), tf.assign(state_value_network.b2, p_b2),
                  tf.assign(state_value_network.W3, p_W3), tf.assign(state_value_network.b3, p_b3)])
        train_mountain_car_model(policy, state_value_network, net_params, algo_params, LOGS_PATH, bin_values)

