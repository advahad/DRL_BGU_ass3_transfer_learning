from configuration import mountain_car_config as cnf
from actorcritic import actor_critic_base_algorithm

actor_critic_base_algorithm.run_actor_critic(cnf, True)
