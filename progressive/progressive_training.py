from configuration import cartpole_config as cnf_mountain_car
from configuration import acrobot_config as cnf_acrobot
from configuration import cartpole_config as cnf_cartpole
from networks.policy_network import PolicyNetwork
from networks.critic_network import StateValueNetwork
from transfer.transfer_logic_base import copy_weights_critic, copy_weights_policy

import tensorflow as tf
from actorcritic.actor_critic_base_algorithm import get_algo_params, get_network_params
import gym

# target domain
logs_path = cnf_cartpole.paths['logs'] + '/transfer'
env = gym.make(cnf_cartpole.env['name'])
env._max_episode_steps = None

# src 1
col1_domain_model_path = cnf_acrobot.paths['model'] + '.meta'
col1_domain_ckp_path = cnf_acrobot.paths['checkpoint']
col1_gym_name = cnf_acrobot.env['name']
# col1_policy_net_scope = 'policy_network_' + col1_gym_name
# col1_critic_net_scope = 'critic_network_' + col1_gym_name
col1_policy_net_scope = 'policy_network'
col1_critic_net_scope = 'critic_network'

# src 2
col2_domain_model_path = cnf_mountain_car.paths['model'] + '.meta'
col2_domain_ckp_path = cnf_mountain_car.paths['checkpoint']
col2_gym_name = cnf_mountain_car.env['name']
# col2_policy_net_scope = 'policy_network_' + col2_gym_name
# col2_critic_net_scope = 'critic_network_' + col2_gym_name
col2_policy_net_scope = 'policy_network'
col2_critic_net_scope = 'critic_network'

# algo_params = get_algo_params(cnf, env)
# net_params = get_network_params(cnf)

tf.reset_default_graph()

# CONFIGURATIONS
POLICY_NET_LAYER_SIZE = 20


class SecondColumnNetwork:
    def __init__(self, state_size, action_size, name='progressive_col2_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.A = tf.placeholder(tf.float32, name="advantage")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            # hidden layer 1
            self.W1 = tf.get_variable("W1", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)

            # hidden layer 2
            self.C1_W2 = tf.placeholder(tf.float32, name="C1_W2")
            self.C1_H1 = tf.placeholder(tf.float32, name="C1_H1")
            self.C1_1 = tf.matmul(self.C1_H1, self.C1_W2, name="C1_1")

            self.W2 = tf.get_variable("W2", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z2 = tf.add(tf.add(tf.matmul(self.A1, self.W2), self.b2), self.C1_1)
            self.A2 = tf.nn.relu(self.Z2)

            # output layer
            self.C1_W3 = tf.placeholder(tf.float32, name="C1_W3")
            self.C1_H2 = tf.placeholder(tf.float32, name="C1_H2")
            self.C1_2 = tf.matmul(self.C1_H2, self.C1_W3, name="C1_2")

            self.W3 = tf.get_variable("W3", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z3 = tf.add(tf.add(tf.matmul(self.A2, self.W3), self.b3), self.C1_2)
            self.output = tf.nn.relu(self.Z3)  # A3

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output), name='actions_distribution')
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                        labels=self.action)  # (y_hat, y)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A, name='loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(
                self.loss)


class ThirdColumnNetwork:
    def __init__(self, state_size, action_size, name='progressive_target_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.A = tf.placeholder(tf.float32, name="advantage")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")

            # hidden layer 1
            self.W1 = tf.get_variable("W1", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)

            # hidden layer 2
            # column 1 contribution
            self.C1_W2 = tf.placeholder(tf.float32, name="C1_W2")
            self.C1_H1 = tf.placeholder(tf.float32, name="C1_H1")
            self.C1_1 = tf.matmul(self.C1_H1, self.C1_W2)

            # column 2 contribution
            self.C2_W2 = tf.placeholder(tf.float32, name="C2_W2")
            self.C2_H1 = tf.placeholder(tf.float32, name="C2_H1")
            self.C2_1 = tf.matmul(self.C2_H1, self.C2_W2)

            self.W2 = tf.get_variable("W2", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z2 = tf.add(tf.add(tf.add(tf.matmul(self.A1, self.W2), self.b2), self.C1_1), self.C2_1)
            self.A2 = tf.nn.relu(self.Z2)

            # output layer
            # column 1 contribution
            self.C1_W3 = tf.placeholder(tf.float32, name="C1_W3")
            self.C1_H2 = tf.placeholder(tf.float32, name="C1_H2")
            self.C1_2 = tf.matmul(self.C1_H2, self.C1_W3, name="C1_2")

            # column 2 contribution
            self.C2_W3 = tf.placeholder(tf.float32, name="C2_W3")
            self.C2_H2 = tf.placeholder(tf.float32, name="C2_H2")
            self.C2_2 = tf.matmul(self.C2_H2, self.C2_W3, name="C2_2")  # C2_2: column 2 hidden 2

            self.W3 = tf.get_variable("W3", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.Z3 = tf.add(tf.add(tf.add(tf.matmul(self.A2, self.W3), self.b3), self.C1_2), self.C2_2)
            self.output = tf.nn.relu(self.Z3)  # A3

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output), name='actions_distribution')
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                        labels=self.action)  # (y_hat, y)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A, name='loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(
                self.loss)


#
graph1 = tf.Graph()
sess1 = tf.Session(graph=graph1)
with graph1.as_default():
    saver = tf.train.import_meta_graph(col1_domain_model_path)
    saver.restore(sess1, tf.train.latest_checkpoint(col1_domain_ckp_path))
    p_col1_W1 = graph1.get_tensor_by_name("policy_network/W1:0")
    print(p_col1_W1)


graph2 = tf.Graph()
sess2 = tf.Session(graph=graph2)
with graph2.as_default():
    saver = tf.train.import_meta_graph(col2_domain_model_path)
    saver.restore(sess2, tf.train.latest_checkpoint(col2_domain_ckp_path))
    p_col2_W1 = graph2.get_tensor_by_name("policy_network/W1:0")
    print(p_col2_W1)

col1_policy = PolicyNetwork(6, 10, name="policy_network_new")
col2_policy = SecondColumnNetwork(6, 10)
target_policy = ThirdColumnNetwork(6, 10)

col1_policy = copy_weights_policy(sess1, graph1, col1_policy)
# need to run feed dict to get W output








