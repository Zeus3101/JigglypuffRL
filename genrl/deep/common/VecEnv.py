import gym
import numpy as np
import multiprocessing as mp

from abc import ABC, abstractmethod


def worker(parent_conn: mp.Pipe, child_conn: mp.Pipe, env: gym.Env):
    """
    Worker class to facilitate multiprocessing

    :param parent_conn: Parent connection of Pipe
    :param child_conn: Child connection of Pipe
    :param env: Gym environment we need multiprocessing for
    :type parent_conn: Multiprocessing Pipe Connection
    :type child_conn: Multiprocessing Pipe Connection
    :type env: Gym Environment
    """
    parent_conn.close()
    while True:
        cmd, data = child_conn.recv()
        if cmd == "step":
            observation, reward, done, info = env.step(data)
            child_conn.send((observation, reward, done, info))
        elif cmd == "seed":
            child_conn.send(env.seed(data))
        elif cmd == "reset":
            observation = env.reset()
            child_conn.send(observation)
        elif cmd == "render":
            child_conn.send(env.render())
        elif cmd == "close":
            env.close()
            child_conn.close()
            break
        elif cmd == "get_spaces":
            child_conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


def create_envs(env_name, n_envs):
    """
    Helper function to return list of environments

    :param env_name: Name of environment to be vectorised
    :param n_envs: Number of environments per VecEnv
    :type env_name: string
    :type n_envs: int
    :returns: Multiple of the same gym environment
    :rtype: list
    """
    envs = []
    for i in range(n_envs):
        envs.append(env)
    return envs


class VecEnv(ABC):
    """
    Constructs a wrapper for serial execution through envs.

    :param env: Gym environment to be vectorised
    :param n_envs: Number of environments
    :type env: Gym Environment
    :type n_envs: int
    """

    def __init__(self, env, n_envs=2):
        self.envs = create_envs(env, n_envs)
        self._n_envs = len(self.envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def __iter__(self):
        """
        Iterator object to iterate through each environment in vector
        """
        return (env for env in self.envs)

    def sample(self):
        """
        Return samples of actions from each environment
        """
        return [env.action_space.sample() for env in self.envs]

    def action_spaces(self):
        """
        Return action spaces of each environment
        """
        return [env.action_space for env in self.envs]

    def __getitem__(self, index):
        """
        Return environment at the given index
        """
        return self.envs[index]

    def seed(self, seed):
        """
        Set seed for reproducability in all environments
        """
        [env.seed(seed) for env in self.envs]

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @property
    def n_envs(self):
        return self._n_envs


class SerialVecEnv(VecEnv):
    """
    Constructs a wrapper for serial execution through envs.

    :param env: Gym environment to be vectorised
    :param n_envs: Number of environments
    :type env: Gym Environment
    :type n_envs: int
    """

    def __init__(self, envs, n_envs=2):
        super(SerialVecEnv, self).__init__(envs, n_envs)

    def step(self, actions):
        """
        Steps through all envs serially

        :param actions: Actions from the model
        :type actions: Iterable of ints/floats
        """
        states, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[i])
            states.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return states, rewards, dones, infos

    def reset(self):
        """
        Resets all envs
        """
        states = []
        for env in self.envs:
            states.append(env.reset())

        return states

    def close(self):
        """
        Closes all envs
        """
        for env in self.envs:
            env.close()

    def images(self):
        """
        Returns an array of images from each env render
        """
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode="human"):
        """
        Renders all envs in a tiles format similar to baselines

        :param mode: Can either be 'human' or 'rgb_array'. Displays tiled \
images in 'human' and returns tiled images in 'rgb_array'
        :type mode: string
        """
        images = np.asarray(self.images())
        N, H, W, C = images.shape
        newW, newH = int(np.ceil(np.sqrt(W))), int(np.ceil(np.sqrt(H)))
        images = np.array(list(images) + [images[0] * 0 for _ in range(N, newH * newW)])
        out_image = images.reshape(newH, newW, H, W, C)
        out_image = out_image.transpose(0, 2, 1, 3, 4)
        out_image = out_image.reshape(newH * H, newW * W, C)
        if mode == "human":
            import cv2  # noqa

            cv2.imshow("vecenv", out_image[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return out_image
        else:
            raise NotImplementedError


class SubProcessVecEnv(VecEnv):
    """
    Constructs a wrapper for serial execution through envs.

    :param env: Environment Name. Should be registered with OpenAI Gym.
    :param n_envs: Number of environments
    :type env: string
    :type n_envs: int
    """

    def __init__(self, env, n_envs=2):
        super(SubProcessVecEnv, self).__init__(env, n_envs)

        self.procs = []
        self.parent_conns, self.child_conns = zip(
            *[mp.Pipe() for i in range(self._n_envs)]
        )

        for parent_conn, child_conn, env_fn in zip(
            self.parent_conns, self.child_conns, self.envs
        ):
            args = (parent_conn, child_conn, env_fn)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.procs.append(process)
            child_conn.close()

    def get_spaces(self):
        """
        Returns state and action spaces of environments
        """
        self.parent_conns[0].send(("get_spaces", None))
        observation_space, action_space = self.parent_conns[0].recv()
        return (observation_space, action_space)

    def seed(self, seed=None):
        """
        Sets seed for reproducability
        """
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("seed", seed + idx))

        return [parent_conn.recv() for parent_conn in self.parent_conns]

    def reset(self):
        """
        Resets environments

        :returns: States after environment reset
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        obs = [parent_conn.recv() for parent_conn in self.parent_conns]
        return obs

    def step(self, actions):
        """
        Steps through environments serially

        :param actions: Actions from the model
        :type actions: Iterable of ints/floats
        """
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(("step", action))
        self.waiting = True

        result = []
        for parent_conn in self.parent_conns:
            result.append(parent_conn.recv())
        self.waiting = False

        observations, rewards, dones, infos = zip(*result)
        print(observations, rewards, dones)
        return observations, rewards, dones, infos

    def close(self):
        """
        Closes all environments and processes
        """
        if self.waiting:
            for parent_conn in self.parent_conns:
                parent_conn.recv()
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
        for proc in self.procs:
            proc.join()


def venv(env, n_envs, parallel=False):
    """
    Chooses the kind of Vector Environment that is required

    :param env: Gym environment to be vectorised
    :param n_envs: Number of environments
    :param parallel: True if we want environments to run parallely and \
subprocesses, False if we want environments to run serially one after the other
    :returns: Vector Environment
    :rtype: VecEnv
    """
    if parallel:
        return SubProcessVecEnv(env, n_envs)
    else:
        return SerialVecEnv(env, n_envs)
