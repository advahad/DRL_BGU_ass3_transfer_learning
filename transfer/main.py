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


if __name__=='__main__':
    curr_game = 'MountainCarContinuous-v0'

    tf.reset_default_graph()
    # policy = networks[curr_game].PolicyNetwork(net_sate_size, net_action_size)
    # baseline=networks[curr_game].BaselineNetwork(net_sate_size)

    with tf.Session() as sess:
        ckpt_path="../models/acrobot/Acrobot-v1-model.ckpt"

        saver = tf.train.import_meta_graph("../models/acrobot/Acrobot-v1-model.ckpt.meta")
        saver.restore(sess, tf.train.latest_checkpoint('../models/acrobot'))
        graph = tf.get_default_graph()
        W3 = graph.get_tensor_by_name("W3_p:0")
        print(W3)
        # tf.get_variable('b4', initializer=tf.zeros_initializer)

        # fit_models(curr_game, policy, baseline, **params[curr_game])
