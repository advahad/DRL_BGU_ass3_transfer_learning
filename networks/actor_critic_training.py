import tensorflow as tf
import numpy as np

from utils import padding_util, summary_util


class NetworkParams:
    def __init__(self, max_state_size, action_size, max_action_size):
        self.max_state_size = max_state_size
        self.action_size = action_size
        self.max_action_size = max_action_size



class AlgorithmParams:
    def __init__(self, env, render, max_episodes, max_steps, discount_factor, policy_learning_rate, learning_rate_decay, solved_th):
        self.env = env
        self.render = render
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.solved_th = solved_th


def decay_learning_rate(learning_rate, episode, learning_rate_decay):
    return max(0.0001, learning_rate * learning_rate_decay ** episode)


def train_models(policy, state_value_network, network_params, algorithm_params, LOGS_PATH, MODEL_PATH):
    saver = tf.train.Saver()
    summary_writer = summary_util.init(LOGS_PATH)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        episode_rewards = np.zeros(algorithm_params.max_episodes)
        average_rewards = 0.0

        for episode in range(algorithm_params.max_episodes):
            state = algorithm_params.env.reset()
            state = padding_util.pad_and_reshape(state, network_params.max_state_size)
            policy_losses = []
            value_losses = []

            i = 1.0
            for step in range(algorithm_params.max_steps):
                # choose action from policy network given initial state

                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                # normal choosing
                # action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

                # choose only existing actions
                actions_distribution = actions_distribution[:network_params.action_size]
                actions_distribution /= actions_distribution.sum()
                action = np.random.choice(np.arange(len(actions_distribution))[:network_params.action_size], p=actions_distribution)


                next_state, reward, done, _ = algorithm_params.env.step(action)
                next_state = padding_util.pad_and_reshape(next_state, network_params.max_state_size)

                if algorithm_params.render:
                    algorithm_params.env.render()

                action_one_hot = np.zeros(network_params.max_action_size)
                action_one_hot[action] = 1

                # update statistics
                episode_rewards[episode] += reward

                # calc advantage
                V_s = sess.run(state_value_network.value_estimate, {state_value_network.state: state})

                if not done:
                    V_s_prime = sess.run(state_value_network.value_estimate, {state_value_network.state: next_state})

                else:
                    V_s_prime = 0

                td_target = reward + algorithm_params.discount_factor * V_s_prime
                td_error = td_target - V_s  # the TD error is the advantage

                # update V network
                state_value_feed_dict = {state_value_network.state: state,
                                         state_value_network.td_target: td_target}
                _, state_value_loss = sess.run([state_value_network.optimizer, state_value_network.loss],
                                               state_value_feed_dict)
                value_losses.append(state_value_loss)

                algorithm_params.policy_learning_rate = decay_learning_rate(algorithm_params.policy_learning_rate, episode, algorithm_params.learning_rate_decay)

                # update policy network
                feed_dict = {policy.state: state,
                             policy.A: td_error * i,
                             policy.action: action_one_hot,
                             policy.learning_rate: algorithm_params.policy_learning_rate}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy_losses.append(loss)

                if done:  # episode done
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    else:
                        average_rewards = np.mean(episode_rewards[0:episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards > algorithm_params.solved_th:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break

                # re-assign
                i = algorithm_params.discount_factor * i
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

        # save the model
        saver.save(sess, MODEL_PATH)
