# import tensorflow as tf
# # saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     with tf.variable_scope('policy_network', reuse=True) as myscope:
#         # saver = tf.train.Saver()
#         # saver.restore(sess, ckpt_path)
#         # tf.get_variable('W4', initializer=tf.contrib.layers.xavier_initializer(seed=0))
#         # tf.get_variable('b4', initializer=tf.zeros_initializer)
#         myscope.run(tf.global_variables_initializer())
#         saver = tf.train.import_meta_graph("../models/acrobot/Acrobot-v1-model.ckpt.meta")
#         model = saver.restore(myscope, "../models/acrobot/Acrobot-v1-model.ckpt")
#         # graph = tf.get_default_graph()
#         # W1_1 = graph.get_tensor_by_name('W3')
#         W4 = tf.get_variable('W3', initializer=tf.contrib.layers.xavier_initializer(seed=0))
#         b4 = tf.get_variable('b3', initializer=tf.zeros_initializer)
#         print("restored")
import tensorflow as tf
from networks.actor_critic_training import train_models
from networks.policy_network import PolicyNetworkSlim
from networks.critic_network import StateValueNetworkSlim
from networks.actor_critic_training import AlgorithmParams, NetworkParams
import numpy as np
import gym

env = gym.make('CartPole-v1')
env._max_episode_steps = None
solved_th = 475

np.random.seed(1)

# CONFIGURATIONS
V_NET_LAYER_SIZE = 20
POLICY_NET_LAYER_SIZE = 20

LOGS_PATH = '../logs/actor-critic/transfer/CartPole-v1'
MODEL_PATH = '../models/cartpole/transfer/CartPole-v1-model'

# Define hyper parameters
state_size = 4
action_size = env.action_space.n
max_state_size = 6
max_action_size = 10

max_episodes = 5000
max_steps = 10000000
discount_factor = 0.99
policy_learning_rate = 0.001
value_net_learning_rate = 0.001
learning_rate_decay = 0.99

render = False

algo_params = AlgorithmParams(env, render, max_episodes, max_steps, discount_factor, policy_learning_rate, learning_rate_decay, solved_th)

net_params = NetworkParams(max_state_size, action_size, max_action_size)


if __name__=='__main__':
    tf.reset_default_graph()
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph("../models/acrobot/Acrobot-v1-model.meta")
        saver.restore(sess, tf.train.latest_checkpoint('../models/acrobot'))
        graph = tf.get_default_graph()
        state = graph.get_tensor_by_name("policy_network/state:0")
        action = graph.get_tensor_by_name("policy_network/action:0")
        A = graph.get_tensor_by_name("policy_network/advantage:0")
        actions_distribution = graph.get_tensor_by_name("policy_network/actions_distribution:0")
        learning_rate = graph.get_tensor_by_name("policy_network/learning_rate:0")
        optimizer = graph.get_operation_by_name("policy_network/optimizer")
        loss = graph.get_tensor_by_name("policy_network/loss:0")

        policy = PolicyNetworkSlim(actions_distribution, state, A, action, learning_rate, optimizer, loss)

        # value_estimate, state, td_target, optimizer, loss)
        value_estimate = graph.get_tensor_by_name("state_value_network/value_estimate:0")
        v_state = graph.get_tensor_by_name("state_value_network/state:0")
        td_target = graph.get_tensor_by_name("state_value_network/td_target:0")
        v_optimizer = graph.get_operation_by_name("state_value_network/optimizer")
        v_loss = graph.get_tensor_by_name("state_value_network/loss:0")
        state_value_net = StateValueNetworkSlim(value_estimate, v_state, td_target, v_optimizer, v_loss)

        train_models(policy, state_value_net, net_params, algo_params, LOGS_PATH, MODEL_PATH)
            # tf.get_variable('b4', initializer=tf.zeros_initializer)

            # fit_models(curr_game, policy, baseline, **params[curr_game])
