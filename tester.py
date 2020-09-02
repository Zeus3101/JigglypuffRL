import sys

import numpy as np

from contrib.maa2c.a2c_agent import A2CAgent
from contrib.maa2c.example.simple_spread_test import make_env
from contrib.maa2c.example.utils import TensorboardLogger
from genrl.environments.suite import MultiEnv

sys.path.append("add path to /genrl/genrl")  # add path


def train(max_episodes, timesteps_per_eps, logdir):
    tensorboard_logger = TensorboardLogger(logdir)
    env = MultiEnv("simple_spread")
    a2c_agent = A2CAgent(env, 2e-4, 0.99, None, 0.008, timesteps_per_eps)

    for episode in range(1, max_episodes + 1):
        print("EPISODE", episode)
        action = np.array(env.sample())
        print(action.shape)
        states = np.asarray(env.reset())
        # print(states.shape)  # Supposed to be (4, 12). It's currently showing (4, 36) where 36 = 24 + 12 after concatenation
        values, dones = a2c_agent.collect_rollouts(states)
        a2c_agent.get_traj_loss(values, dones)
        a2c_agent.update_policy()
        params = a2c_agent.get_logging_params()
        params["episode"] = episode
        tensorboard_logger.write(params)
    tensorboard_logger.close()


if __name__ == "__main__":
    train(max_episodes=1000, timesteps_per_eps=300, logdir="./runs/")
