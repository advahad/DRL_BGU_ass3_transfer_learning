import tensorflow as tf

# CONFIGURATIONS
V_NET_LAYER_SIZE = 20


class StateValueNetwork:
    def __init__(self, state_size, output_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.td_target = tf.placeholder(tf.float32, name="td_target")

            self.W1 = tf.get_variable("W1", [self.state_size, V_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [V_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [V_NET_LAYER_SIZE, V_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [V_NET_LAYER_SIZE], initializer=tf.zeros_initializer())

            self.W3 = tf.get_variable("W3", [V_NET_LAYER_SIZE, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.Z3 = tf.add(tf.matmul(self.A2, self.W3), self.b3)
            self.value_estimate = tf.squeeze(self.Z3, name='value_estimate')

            self.loss = tf.squared_difference(self.value_estimate, self.td_target, name='loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(self.loss)


class StateValueNetworkSlim:

    def __init__(self, value_estimate, state, td_target, optimizer, loss):
        self.value_estimate = value_estimate
        self.state = state
        self.td_target = td_target
        self.optimizer = optimizer
        self.loss = loss
