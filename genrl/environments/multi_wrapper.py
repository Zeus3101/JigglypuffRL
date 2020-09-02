from typing import Any

import gym
import numpy as np

from genrl.environments import GymWrapper


class MultiWrapper(GymWrapper):
    def __init__(self, env: gym.Env):
        super(MultiWrapper, self).__init__(env)
        self.env = env

    def __getattr__(self, name: str) -> Any:
        env = super(MultiWrapper, self).__getattribute__("env")
        return getattr(env, name)

    def sample(self):
        return [self.env.action_space[i].sample() for i in range(self.env.n)]

    def step(self, action: np.ndarray) -> np.ndarray:
        state, self.reward, self.done, self.info = self.env.step(action)
        self.action = action
        self.state = np.stack([np.concatenate(s) for s in state])
        return self.state, self.reward, self.done, self.info

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        self.state = np.stack([np.concatenate(s) for s in state])
        return self.state
