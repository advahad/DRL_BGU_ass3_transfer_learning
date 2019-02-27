import gym
import numpy as np

from actorcritic.actor_critic_base_algorithm import get_algo_params, get_network_params
from configuration import mountain_car_config as cnf
from transfer.transfer_logic_base import transfer_and_train

env = gym.make(cnf.env['name'])
env._max_episode_steps = 7000
env.seed(1)
np.random.seed(1)

algo_params = get_algo_params(cnf, env)

net_params = get_network_params(cnf)
src_domain_model_path = "../models/cartpole/cartpole-model.meta"
src_domain_ckp_path = '../models/cartpole'

transfer_and_train(src_domain_model_path, src_domain_ckp_path, net_params, algo_params, cnf, True)
