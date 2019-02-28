import tensorflow as tf
from networks.policy_network import PolicyNetwork
from networks.critic_network import StateValueNetwork
from networks.training import train_model, train_mountain_car

bin_values = [-0.7, -0.6, -0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7]


def transfer_and_train(src_domain_model_path, src_domain_ckp_path, net_params, algo_params, cnf, is_mountain_car=False):
    logs_path = cnf.paths['logs'] + '/transfer'

    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(src_domain_model_path)
        saver.restore(sess, tf.train.latest_checkpoint(src_domain_ckp_path))
        graph = tf.get_default_graph()

        policy = PolicyNetwork(net_params.max_state_size, net_params.max_action_size, name="policy_network_new")
        state_value_network = StateValueNetwork(net_params.max_state_size, 1, algo_params.value_net_learning_rate,
                                                name='state_value_net_new')

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
        if not is_mountain_car:
            train_model(policy, state_value_network, net_params, algo_params, logs_path)
        else:
            train_model(policy, state_value_network, net_params, algo_params, logs_path, True, bin_values)
