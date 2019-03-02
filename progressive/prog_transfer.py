import tensorflow as tf
import os
import gym
from progressive.progressive_training import ProgNetPolicy, ProgNetValue
from networks.training import train_model, NetworkParams, AlgorithmParams
from actorcritic.actor_critic_base_algorithm import get_algo_params, get_network_params
from configuration import cartpole_config, mountain_car_config, acrobot_config


def get_pol_and_val_network_params(src_domain_model_path):
    my_graph = tf.Graph()
    with my_graph.as_default():
        with tf.Session(graph=my_graph).as_default() as sess:
            saver = tf.train.import_meta_graph(src_domain_model_path + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(src_domain_model_path)))
            pol_w1 = my_graph.get_tensor_by_name("policy_network/W1:0")
            pol_w2 = my_graph.get_tensor_by_name("policy_network/W2:0")
            pol_b1 = my_graph.get_tensor_by_name("policy_network/b1:0")
            pol_b2 = my_graph.get_tensor_by_name("policy_network/b2:0")

            val_w1 = my_graph.get_tensor_by_name("state_value_network/W1:0")
            val_w2 = my_graph.get_tensor_by_name("state_value_network/W2:0")
            val_b1 = my_graph.get_tensor_by_name("state_value_network/b1:0")
            val_b2 = my_graph.get_tensor_by_name("state_value_network/b2:0")

            pol_w1, pol_w2, pol_b1, pol_b2, val_w1, val_w2, val_b1, val_b2 =\
                sess.run([pol_w1, pol_w2, pol_b1, pol_b2, val_w1, val_w2, val_b1, val_b2])
    return pol_w1, pol_w2, pol_b1, pol_b2, val_w1, val_w2, val_b1, val_b2


def train_prograssive(path_net1, path_net2, cnf):
    pol1_w1, pol1_w2, pol1_b1, pol1_b2, val1_w1, val1_w2, val1_b1, val1_b2 = get_pol_and_val_network_params(os.path.abspath(path_net1))
    pol2_w1, pol2_w2, pol2_b1, pol2_b2, val2_w1, val2_w2, val2_b1, va2_b2 = get_pol_and_val_network_params(os.path.abspath(path_net2))

    my_graph = tf.Graph()
    with my_graph.as_default():
        with tf.Session(graph=my_graph).as_default() as sess:
            policy = ProgNetPolicy(cnf.network['max_state_size'], cnf.network['max_action_size'])
            value_net = ProgNetValue(cnf.network['max_state_size'], 1, cnf.network['value_net_learning_rate'])
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            col1_policy_w1 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col1/W11:0')
            col1_policy_w2 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col1/W21:0')
            col1_policy_b1 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col1/b11:0')
            col1_policy_b2 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col1/b21:0')
            col1_value_w1 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col1/W11:0')
            col1_value_w2 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col1/W21:0')
            col1_value_b1 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col1/b11:0')
            col1_value_b2 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col1/b21:0')
            col2_policy_w1 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col2/W12:0')
            col2_policy_w2 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col2/W22:0')
            col2_policy_b1 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col2/b12:0')
            col2_policy_b2 = my_graph.get_tensor_by_name('progressive_policy_network/prog_net_col2/b22:0')
            col2_value_w1 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col2/W12:0')
            col2_value_w2 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col2/W22:0')
            col2_value_b1 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col2/b12:0')
            col2_value_b2 = my_graph.get_tensor_by_name('progressive_state_value_network/prog_net_col2/b22:0')

            sess.run([tf.assign(col1_policy_w1, pol1_w1), tf.assign(col1_policy_w2, pol1_w2)])
            sess.run([tf.assign(col1_policy_b1, pol1_b1), tf.assign(col1_policy_b2, pol1_b2)])
            sess.run([tf.assign(col1_value_w1, val1_w1), tf.assign(col1_value_w2, val1_w2)])
            sess.run([tf.assign(col1_value_b1, val1_b1), tf.assign(col1_value_b2, val1_b2)])
            sess.run([tf.assign(col2_policy_w1, pol2_w1), tf.assign(col2_policy_w2, pol2_w2)])
            sess.run([tf.assign(col2_policy_b1, pol2_b1), tf.assign(col2_policy_b2, pol2_b2)])
            sess.run([tf.assign(col2_value_w1, val2_w1), tf.assign(col2_value_w2, val2_w2)])
            sess.run([tf.assign(col2_value_b1, val2_b1), tf.assign(col2_value_b2, va2_b2)])

            policy.set_network_number(3)
            value_net.set_network_number(3)

            env = gym.make(cnf.env['name'])
            env._max_episode_steps = cnf.env['max_episode_steps']
            algo_params = get_algo_params(cnf, env)
            net_params = get_network_params(cnf)
            logs_path = cnf.paths['logs'] + '/progressive'

            train_model(policy, value_net, net_params, algo_params, logs_path, mode="progressive", is_mountain_car=True, bin_values=cnf.bin_values)


if __name__ == '__main__':
    conf1 = [acrobot_config.paths['model'], mountain_car_config.paths['model'], cartpole_config]
    conf2 = [cartpole_config.paths['model'], acrobot_config.paths['model'], mountain_car_config]

    train_prograssive(*conf2)

