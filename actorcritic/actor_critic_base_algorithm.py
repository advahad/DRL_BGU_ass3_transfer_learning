import gym
import numpy as np
import tensorflow as tf

from networks.actor_critic_training import train_model, train_mountain_car, NetworkParams, \
    AlgorithmParams
from networks.critic_network import StateValueNetwork
from networks.policy_network import PolicyNetwork

bin_values = [-0.7, -0.6, -0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7]

np.random.seed(1)


def get_network_params(cnf):
    network_params = NetworkParams(cnf.network['max_state_size'], cnf.env['action_size'],
                                   cnf.network['max_action_size'])
    return network_params


def get_algo_params(cnf, env):
    algo_params = AlgorithmParams(env, cnf.env['render'], cnf.algo['max_episodes'], cnf.algo['max_steps'],
                                  cnf.algo['discount_factor'], cnf.network['policy_learning_rate'],
                                  cnf.network['value_net_learning_rate'], cnf.network['learning_rate_decay'],
                                  cnf.env['solved_th'])
    return algo_params


def generate_policy_net(cnf):
    policy = PolicyNetwork(cnf.network['max_state_size'], cnf.network['max_action_size'])
    return policy


def generate_critic_net(cnf):
    state_value_network = StateValueNetwork(cnf.network['max_state_size'], 1, cnf.network['value_net_learning_rate'])
    return state_value_network


def run_actor_critic(cnf, is_mountain_car):
    env = gym.make(cnf.env['name'])
    env.seed(1)
    if is_mountain_car:
        env._max_episode_steps = 7000
    else:
        env._max_episode_steps = None
    np.random.seed(1)

    tf.reset_default_graph()
    policy = PolicyNetwork(cnf.network['max_state_size'], cnf.network['max_action_size'])
    state_value_network = StateValueNetwork(cnf.network['max_state_size'], 1, cnf.network['value_net_learning_rate'])

    # params initializations
    network_params = NetworkParams(cnf.network['max_state_size'], cnf.env['action_size'],
                                   cnf.network['max_action_size'])
    algo_params = AlgorithmParams(env, cnf.env['render'], cnf.algo['max_episodes'], cnf.algo['max_steps'],
                                  cnf.algo['discount_factor'], cnf.network['policy_learning_rate'],
                                  cnf.network['value_net_learning_rate'],
                                  cnf.network['learning_rate_decay'], cnf.env['solved_th'])

    # start training
    if not is_mountain_car:
        train_model(policy, state_value_network, network_params, algo_params,
                    cnf.paths['logs'] + '/baseline', cnf.paths['model'], save_model=True)
    else:
        train_mountain_car(policy, state_value_network, network_params, algo_params, bin_values,
                           cnf.paths['logs'] + '/baseline', cnf.paths['model'], save_model=True)
    # train_models(policy, state_value_network, network_params, algo_params,
    #              cnf.paths['logs'] + '/baseline', cnf.paths['model'], save_model=True)
