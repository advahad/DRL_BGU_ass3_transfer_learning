import gym
import numpy as np
import tensorflow as tf
import summary_util
import bins_container

# env setup
env = gym.make('MountainCarContinuous-v0')
env._max_episode_steps = 7000
env.seed(1)
np.random.seed(1)

# CONFIGURATIONS
V_NET_LAYER_SIZE = 20
POLICY_NET_LAYER_SIZE = 20 # may be changed to 12

LOGS_PATH = './logs/actor-critic/MountainCarContinuous-v0'

# Define hyper parameters
state_size = 2
action_size = 10
# action_size = env.action_space.n
max_state_size = 6

max_episodes = 5000
max_steps = 10000000
discount_factor = 0.99
policy_learning_rate = 0.0001
value_net_learning_rate = 0.001
learning_rate_decay = 0.995 # may be 0.999
solved_th = 80

bin_values = [-0.7,-0.6,-0.5,-0.4,-0.2,0,0.2,0.4,0.6,0.7]

render = False


# discretizing states
car_position_range = [env.observation_space.low[0], env.observation_space.high[0]]
car_velocity_range = [env.observation_space.low[1], env.observation_space.high[1]]

action_range = [env.action_space.low[0], env.action_space.high[0]]

car_position_discrete = np.linspace(car_position_range[0], car_position_range[1], 21)
car_velocity_discrete = np.linspace(car_velocity_range[0], car_velocity_range[1], 21)
action_discrete = np.linspace(action_range[0], action_range[1], 21)

car_pos_bins = bins_container.DiscreteBins(car_position_discrete)
car_velocity_bins = bins_container.DiscreteBins(action_discrete)
action_bins = bins_container.DiscreteBins(action_discrete)


def get_idx(action_cont):
    for i in range(len(bin_values)):
        if action_cont < bin_values[i]:
            return i
    return len(bin_values) - 1

def pad_state(state):
    pad = np.zeros(max_state_size - len(state))
    np.concatenate((state, pad), axis=0)


def decay_learning_rate(learning_rate, episode):
    return max(0.0001, learning_rate * learning_rate_decay ** episode)


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
            self.value_estimate = tf.squeeze(self.Z3)

            self.loss = tf.squared_difference(self.value_estimate, self.td_target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


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
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                        labels=self.action)  # (y_hat, y)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Initialize the policy network
tf.reset_default_graph()

policy = PolicyNetwork(state_size, action_size)
state_value_network = StateValueNetwork(state_size, 1, value_net_learning_rate)

summary_writer = summary_util.init(LOGS_PATH)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    for episode in range(max_episodes):
        print("Starting episode: %d" % episode)
        state = env.reset()
        # padding
        # state = np.append(state, [0] * (state_size - len(state)))
        state = state.reshape([1, state_size])
        policy_losses = []
        value_losses = []

        i = 1.0
        for step in range(max_steps):
            # choose action from policy network given initial state
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(bin_values, 1, p=actions_distribution)
            #cont_action = get_action_cont(action)
            #summaried_action = cont_action + state[0][1]
            summaried_action = action + state[0][1]
            next_state, reward, done, _ = env.step(summaried_action)

            #pad
            #next_state = np.append(next_state, [0] * (state_size - len(next_state)))
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)

            action_one_hot[get_idx(summaried_action)] = 1#get_action_discrete(action)

            # update cont_actionstatistics
            episode_rewards[episode] += reward

            # calc advantage
            V_s = sess.run(state_value_network.value_estimate, {state_value_network.state: state})

            if not done:
                V_s_prime = sess.run(state_value_network.value_estimate,
                                     {state_value_network.state: next_state})

            else:
                V_s_prime = 0

            td_target = reward + discount_factor * V_s_prime
            td_error = td_target - V_s  # the TD error is the advantage

            # update V network
            state_value_feed_dict = {state_value_network.state: state,
                                     state_value_network.td_target: td_target}
            _, state_value_loss = sess.run([state_value_network.optimizer, state_value_network.loss],
                                           state_value_feed_dict)
            value_losses.append(state_value_loss)

            policy_learning_rate = decay_learning_rate(policy_learning_rate, episode)

            # update policy network
            feed_dict = {policy.state: state,
                         policy.A: td_error * i,
                         policy.action: action_one_hot,
                         policy.learning_rate: policy_learning_rate}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            policy_losses.append(loss)

            if done:  # episode done
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[0:episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > solved_th:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            # re-assign
            i = discount_factor * i
            state = next_state

        # update and save tensorboared summaries
        policy_episode_summary = summary_util.create_avg_summary(policy_losses, "policy loss")
        value_episode_summary = summary_util.create_avg_summary(value_losses, "value loss")
        rewards_summary = summary_util.create_summary(episode_rewards[episode], "total rewards")
        summaries = [policy_episode_summary, value_episode_summary, rewards_summary]
        summary_util.write_summaries(summary_writer, episode, summaries)

        if solved:
            break

    summary_writer.close()
