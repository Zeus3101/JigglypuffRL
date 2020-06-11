from gym import spaces
import torch
import numpy as np

from .base import BaseActorCritic
from .policies import MlpPolicy
from .values import MlpValue
from .utils import cnn


class MlpActorCritic(BaseActorCritic):
    """
    MLP Actor Critic

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param val_type: Specifies type of value function: \
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type val_type: str
    :type discrete: bool
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        hidden: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        *args,
        **kwargs
    ):
        super(MlpActorCritic, self).__init__()

        self.actor = MlpPolicy(state_dim, action_dim, hidden, discrete, **kwargs)
        self.critic = MlpValue(state_dim, action_dim, val_type, hidden)


class CNNActorCritic(BaseActorCritic):
    """
    CNN Actor Critic

    :param framestack: Number of previous frames to stack together
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param val_type: Specifies type of value function: (
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a))
    :param discrete: True if action space is discrete, else False
    :type framestack: int
    :type action_dim: int
    :type hidden: tuple or list
    :type val_type: str
    :type discrete: bool
    """
    def __init__(
        self,
        framestack: int,
        action_dim: spaces.Space,
        hidden: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        *args,
        **kwargs
    ):
        super(CNNActorCritic, self).__init__()

        self.conv, output_size = cnn((framestack, 16, 32))
        self.actor = MlpPolicy(output_size, action_dim, hidden, discrete, **kwargs)
        self.critic = MlpValue(output_size, action_dim, val_type, hidden)
        print(output_size)

    def feature(self, state: torch.Tensor):
        """
        Feature extractor method to pass before using get_action and get_value
        """
        phi_state = self.conv(state)
        phi_state = state.view(state.size(0), -1)
        return phi_state

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get action from the Actor based on input

        :param state: The state being passed as input to the Actor
        :param deterministic: (True if the action space is deterministic,
else False)
        :type state: Tensor
        :type deterministic: boolean
        :returns: action
        """
        state = torch.as_tensor(state).float()
        phi_state = self.feature(state)
        print(state.shape, phi_state.shape)
        return self.actor.get_action(phi_state, deterministic=deterministic)

    def get_value(self, state: np.ndarray) -> torch.Tensor:
        """
        Get value from the Critic based on input

        :param state: Input to the Critic
        :type state: Tensor
        :returns: value
        """
        state = torch.as_tensor(state).float()
        phi_state = self.feature(state)
        return self.critic.get_value(phi_state)


actor_critic_registry = {"mlp": MlpActorCritic, "cnn": CNNActorCritic}


def get_actor_critic_from_name(name_: str):
    """
    Returns Actor Critic given the type of the Actor Critic

    :param ac_name: Name of the policy needed
    :type ac_name: str
    :returns: Actor Critic class to be used
    """
    if name_ in actor_critic_registry:
        return actor_critic_registry[name_]
    raise NotImplementedError
