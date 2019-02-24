import tensorflow as tf

# CONFIGURATIONS
POLICY_NET_LAYER_SIZE = 20


class PolicyNetwork:
    def __init__(self, state_size, action_size, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.A = tf.placeholder(tf.float32, name="advantage")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.W1 = tf.get_variable("W1", [self.state_size, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [POLICY_NET_LAYER_SIZE, POLICY_NET_LAYER_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [POLICY_NET_LAYER_SIZE], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [POLICY_NET_LAYER_SIZE, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output), name='actions_distribution')
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                        labels=self.action)  # (y_hat, y)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A, name='loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(self.loss)


class PolicyNetworkSlim:
    def __init__(self, actions_distribution, state, A, action, learning_rate, optimizer, loss):
        self.actions_distribution = actions_distribution
        self.state = state
        self.A = A
        self.action = action
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
