from collections import deque
import numpy as np
import gym
from gym.spaces import Box
from gym.core import Wrapper

from typing import List, Tuple

from .vector_envs import VecEnv


class LazyFrames(object):
    """
    Efficient data structure to save each frame only once. \
Can use LZ4 compression to optimize memory usage.

    :param frames: List of frames that needs to converted \
to a LazyFrames data structure
    :param compress: True if we want to use LZ4 compression \
to conserve memory usage
    :type frames: collections.deque
    :type compress: boolean
    """

    def __init__(self, frames: List, compress: bool = False):
        if compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.compress = compress

    def __array__(self) -> np.ndarray:
        """
        Makes the LazyFrames object convertible to a NumPy array
        """
        if self.compress:
            from lz4.block import decompress

            frames = [
                np.frombuffer(decompress(frame), dtype=self._frames[0].dtype).reshape(
                    self._frames[0].shape
                )
                for frame in self._frames
            ]
        else:
            frames = self._frames

        return np.stack(frames, axis=0)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return frame at index
        """
        return self.__array__()[index]

    def __len__(self) -> int:
        """
        Return length of data structure
        """
        return len(self.__array__())

    def __eq__(self, other: np.ndarray) -> bool:
        """
        Compares if data structure is equivalent to another object

        :param other: Other object for comparison
        :type other: object
        """
        return self.__array__() == other

    @property
    def shape(self) -> Tuple:
        """
        Returns dimensions of other object
        """
        return self.__array__().shape


class VecFrameStack(Wrapper):
    """
    VecEnv Wrapper to stack the last few(4 by default) observations of \
agent efficiently

    :param env: Environment to be wrapped
    :param framestack: Number of frames to be stacked
    :param compress: True if we want to use LZ4 compression \
to conserve memory usage
    :type env: Gym Environment
    :type framestack: int
    :type compress: bool
    """

    def __init__(self, venv: VecEnv, framestack: int = 4, compress: bool = False):
        super(FrameStack, self).__init__(env)

        self.venv = venv
        self.n_envs = env.n_envs

        self.framestack = framestack

        low = np.repeat(
            self.env.observation_space.low, framestack, axis=-1
        )
        high = np.repeat(
            self.env.observation_space.high, framestack, axis=-1
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.env.observation_space.dtype
        )

        self._frames = np.zeros((self.n_envs, *low.shape), dtype=low.dtype)

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Steps through environment

        :param action: Action taken by agent
        :type action: NumPy Array
        :returns: Next state, reward, done, info
        :rtype: NumPy Array, float, boolean, dict
        """
        observations, rewards, dones, infos = self.venv.step(action)
        for i, obs in enumerate(observations):
            self._frames[i] = observation
        observations = np.array([self._get_obs[i] for i in range(self.n_envs)])
        return observations, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Resets environment

        :returns: Initial state of environment
        :rtype: NumPy Array
        """
        for i in range(self.n_envs):
            observation = self.env.reset()
            self._frames[i] = np.repeat(observation, self.framestack, axis=-1)
        return self._get_obs()

    def _get_obs(self, i) -> np.ndarray:
        """
        Gets observations given deque of frames

        :returns: Past few frames
        :rtype: NumPy Array
        """
        return np.array(LazyFrames(self._frames[i]))[np.newaxis, ...]
