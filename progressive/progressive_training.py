from configuration import cartpole_config as cnf_mountain_car
from configuration import acrobot_config as cnf_acrobot
from configuration import cartpole_config as cnf_cartpole
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

col1_policy_net_scope = 'policy_network'
col1_critic_net_scope = 'critic_network'




# algo_params = get_algo_params(cnf, env)
# net_params = get_network_params(cnf)

tf.reset_default_graph()

# CONFIGURATIONS
LAYER_SIZE = 20


class ProgNetPolicy:
    def __init__(self, state_size, action_size, name='progressive_policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.current_network_number = 1

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.A = tf.placeholder(tf.float32, name="advantage")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            # Xij - i:depth j:network num
            with tf.variable_scope('prog_net_col1'):
                self.W11 = tf.get_variable("W11", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b11 = tf.get_variable("b11", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W21 = tf.get_variable("W21", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b21 = tf.get_variable("b21", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W31 = tf.get_variable("W31", [LAYER_SIZE, self.action_size],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b31 = tf.get_variable("b31", [self.action_size], initializer=tf.zeros_initializer())

                self.Z11 = tf.add(tf.matmul(self.state, self.W11), self.b11)
                self.A11 = tf.nn.relu(self.Z11)
                self.Z21 = tf.add(tf.matmul(self.A11, self.W21), self.b21)
                self.A21 = tf.nn.relu(self.Z21)
                self.output1 = tf.add(tf.matmul(self.A21, self.W31), self.b31)

                # Softmax probability distribution over actions
                self.actions_distribution1 = tf.squeeze(tf.nn.softmax(self.output1), name='actions_distribution1')
                # Loss with negative log probability
                self.neg_log_prob1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.output1,
                                                                            labels=self.action)  # (y_hat, y)
                self.loss1 = tf.reduce_mean(self.neg_log_prob1 * self.A, name='loss1')
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        name='optimizer1').minimize(self.loss1)

            # uij_wk  from - i:depth j:network num. to - w:depth k:network num
            with tf.variable_scope('prog_net_col2'):
                self.W12 = tf.get_variable("W12", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b12 = tf.get_variable("b12", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W22 = tf.get_variable("W22", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b22 = tf.get_variable("b22", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W32 = tf.get_variable("W32", [LAYER_SIZE, self.action_size],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b32 = tf.get_variable("b32", [self.action_size], initializer=tf.zeros_initializer())

                self.u11_22 = tf.get_variable("u11_22", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u21_32 = tf.get_variable("u21_32", [LAYER_SIZE, self.action_size],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))

                self.Z12 = tf.add(tf.matmul(self.state, self.W12), self.b12)
                self.A12 = tf.nn.relu(self.Z12)
                self.Z22 = tf.add(tf.add(tf.matmul(tf.stop_gradient(self.A11), self.u11_22), tf.matmul(self.A12, self.W22)), self.b22)
                self.A22 = tf.nn.relu(self.Z22)
                self.output2 = tf.add(tf.add(tf.matmul(tf.stop_gradient(self.A21), self.u21_32), tf.matmul(self.A22, self.W32)), self.b32)

                # Softmax probability distribution over actions
                self.actions_distribution2 = tf.squeeze(tf.nn.softmax(self.output2), name='actions_distribution2')
                # Loss with negative log probability
                self.neg_log_prob2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.output2,
                                                                             labels=self.action)  # (y_hat, y)
                self.loss2 = tf.reduce_mean(self.neg_log_prob2 * self.A, name='loss2')
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                         name='optimizer2').minimize(self.loss2)

            with tf.variable_scope('prog_net_col3'):
                self.W13 = tf.get_variable("W13", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b13 = tf.get_variable("b13", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W23 = tf.get_variable("W23", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b23 = tf.get_variable("b23", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W33 = tf.get_variable("W33", [LAYER_SIZE, self.action_size],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b33 = tf.get_variable("b33", [self.action_size], initializer=tf.zeros_initializer())

                self.u11_23 = tf.get_variable("u11_23", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u12_23 = tf.get_variable("u21_23", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u21_33 = tf.get_variable("u21_33", [LAYER_SIZE, self.action_size],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u22_33 = tf.get_variable("u22_33", [LAYER_SIZE, self.action_size],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))

                self.Z13 = tf.add(tf.matmul(self.state, self.W13), self.b13)
                self.A13 = tf.nn.relu(self.Z13)
                self.Z23 = tf.add(
                    tf.add(
                        tf.add(tf.matmul(tf.stop_gradient(self.A11), self.u11_23), tf.matmul(tf.stop_gradient(self.A12), self.u12_23)),
                                  tf.matmul(self.A13, self.W23)), self.b23)
                self.A23 = tf.nn.relu(self.Z23)
                self.output3 = tf.add(
                    tf.add(
                        tf.add(tf.matmul(tf.stop_gradient(self.A21), self.u21_33), tf.matmul(tf.stop_gradient(self.A22), self.u22_33)),
                                  tf.matmul(self.A23, self.W33)), self.b33)


                # Softmax probability distribution over actions
                self.actions_distribution3 = tf.squeeze(tf.nn.softmax(self.output3), name='actions_distribution3')
                # Loss with negative log probability
                self.neg_log_prob3 = tf.nn.softmax_cross_entropy_with_logits(logits=self.output3,
                                                                             labels=self.action)  # (y_hat, y)
                self.loss3 = tf.reduce_mean(self.neg_log_prob3 * self.A, name='loss3')
                self.optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                         name='optimizer3').minimize(self.loss3)

        self.actions_distribution = self.actions_distribution1
        self.loss = self.loss1
        self.optimizer = self.optimizer1


    def set_network_number(self, net_number):
        self.current_network_number = net_number

    def get_network(self):
        if self.current_network_number == 1:
            return self.actions_distribution1, self.optimizer1, self.loss1
        elif self.current_network_number == 2:
            return self.actions_distribution2, self.optimizer2, self.loss2
        else:
            return self.actions_distribution3, self.optimizer3, self.loss3


class ProgNetValue:
    def __init__(self, state_size, output_size, learning_rate, name='progressive_state_value_network'):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.current_network_number = 1

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.td_target = tf.placeholder(tf.float32, name="td_target")

            # Xij - i:depth j:network num
            with tf.variable_scope('prog_net_col1'):
                self.W11 = tf.get_variable("W11", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b11 = tf.get_variable("b11", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W21 = tf.get_variable("W21", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b21 = tf.get_variable("b21", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W31 = tf.get_variable("W31", [LAYER_SIZE, 1],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b31 = tf.get_variable("b31", [1], initializer=tf.zeros_initializer())

                self.Z11 = tf.add(tf.matmul(self.state, self.W11), self.b11)
                self.A11 = tf.nn.relu(self.Z11)
                self.Z21 = tf.add(tf.matmul(self.A11, self.W21), self.b21)
                self.A21 = tf.nn.relu(self.Z21)
                self.output1 = tf.add(tf.matmul(self.A21, self.W31), self.b31)

                self.value_estimate1 = tf.squeeze(self.output1, name='value_estimate1')
                self.loss1 = tf.squared_difference(self.value_estimate1, self.td_target, name='loss1')
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer1').minimize(
                    self.loss1)

            # uij_wk  from - i:depth j:network num. to - w:depth k:network num
            with tf.variable_scope('prog_net_col2'):
                self.W12 = tf.get_variable("W12", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b12 = tf.get_variable("b12", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W22 = tf.get_variable("W22", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b22 = tf.get_variable("b22", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W32 = tf.get_variable("W32", [LAYER_SIZE, 1],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b32 = tf.get_variable("b32", [1], initializer=tf.zeros_initializer())

                self.u11_22 = tf.get_variable("u11_22", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u21_32 = tf.get_variable("u21_32", [LAYER_SIZE, 1],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))

                self.Z12 = tf.add(tf.matmul(self.state, self.W12), self.b12)
                self.A12 = tf.nn.relu(self.Z12)
                self.Z22 = tf.add(tf.add(tf.matmul(tf.stop_gradient(self.A11), self.u11_22), tf.matmul(self.A12, self.W22)), self.b22)
                self.A22 = tf.nn.relu(self.Z22)
                self.output2 = tf.add(tf.add(tf.matmul(tf.stop_gradient(self.A21), self.u21_32), tf.matmul(self.A22, self.W32)), self.b32)

                self.value_estimate2 = tf.squeeze(self.output2, name='value_estimate2')
                self.loss2 = tf.squared_difference(self.value_estimate2, self.td_target, name='loss2')
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer2').minimize(
                    self.loss2)


            with tf.variable_scope('prog_net_col3'):
                self.W13 = tf.get_variable("W13", [self.state_size, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b13 = tf.get_variable("b13", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W23 = tf.get_variable("W23", [LAYER_SIZE, LAYER_SIZE],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b23 = tf.get_variable("b23", [LAYER_SIZE], initializer=tf.zeros_initializer())
                self.W33 = tf.get_variable("W33", [LAYER_SIZE, 1],
                                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.b33 = tf.get_variable("b33", [1], initializer=tf.zeros_initializer())

                self.u11_23 = tf.get_variable("u11_23", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u12_23 = tf.get_variable("u21_23", [LAYER_SIZE, LAYER_SIZE],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u21_33 = tf.get_variable("u21_33", [LAYER_SIZE, 1],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
                self.u22_33 = tf.get_variable("u22_33", [LAYER_SIZE, 1],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))

                self.Z13 = tf.add(tf.matmul(self.state, self.W13), self.b13)
                self.A13 = tf.nn.relu(self.Z13)
                self.Z23 = tf.add(
                    tf.add(
                        tf.add(tf.matmul(tf.stop_gradient(self.A11), self.u11_23), tf.matmul(tf.stop_gradient(self.A12), self.u12_23)),
                                  tf.matmul(self.A13, self.W23)), self.b23)
                self.A23 = tf.nn.relu(self.Z23)
                self.output3 = tf.add(
                    tf.add(
                        tf.add(tf.matmul(tf.stop_gradient(self.A21), self.u21_33), tf.matmul(tf.stop_gradient(self.A22), self.u22_33)),
                                  tf.matmul(self.A23, self.W33)), self.b33)

                self.value_estimate3 = tf.squeeze(self.output3, name='value_estimate3')
                self.loss3 = tf.squared_difference(self.value_estimate3, self.td_target, name='loss3')
                self.optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer3').minimize(
                    self.loss3)

        self.loss = self.loss1
        self.optimizer = self.optimizer1
        self.value_estimate = self.value_estimate1


    def set_network_number(self, net_number):
        self.current_network_number = net_number

        if self.current_network_number == 1:
            self.loss = self.loss1
            self.optimizer = self.optimizer1
            self.value_estimate = self.value_estimate1
        elif self.current_network_number == 2:
            self.loss = self.loss2
            self.optimizer = self.optimizer2
            self.value_estimate = self.value_estimate2
        else:
            self.loss = self.loss3
            self.optimizer = self.optimizer3
            self.value_estimate = self.value_estimate3



    def get_network(self):
        if self.current_network_number == 1:
            return self.actions_distribution1, self.optimizer1, self.loss1
        elif self.current_network_number == 2:
            return self.actions_distribution2, self.optimizer2, self.loss2
        else:
            return self.actions_distribution3, self.optimizer3, self.loss3




#
# #
# graph1 = tf.Graph()
# sess1 = tf.Session(graph=graph1)
# with graph1.as_default():
#     saver = tf.train.import_meta_graph(col1_domain_model_path)
#     saver.restore(sess1, tf.train.latest_checkpoint(col1_domain_ckp_path))
#     p_col1_W1 = graph1.get_tensor_by_name("policy_network/W1:0")
#     print(p_col1_W1)
#
#
# graph2 = tf.Graph()
# sess2 = tf.Session(graph=graph2)
# with graph2.as_default():
#     saver = tf.train.import_meta_graph(col2_domain_model_path)
#     saver.restore(sess2, tf.train.latest_checkpoint(col2_domain_ckp_path))
#     p_col2_W1 = graph2.get_tensor_by_name("policy_network/W1:0")
#     print(p_col2_W1)
#
# col1_policy = PolicyNetwork(6, 10, name="policy_network_new")
# col2_policy = SecondColumnNetwork(6, 10)
# target_policy = ThirdColumnNetwork(6, 10)
#
# col1_policy = copy_weights_policy(sess1, graph1, col1_policy)
# # need to run feed dict to get W output
#
#
#
#
#
#
#

