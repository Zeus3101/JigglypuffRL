import torch
from torch import nn
from .base import BasePolicy
from .utils import mlp, cnn
from typing import Tuple


class MlpPolicy(BasePolicy):
    """
    MLP Policy

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type discrete: bool
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: Tuple = (32, 32),
        discrete: bool = True,
        **kwargs
    ):
        super(MlpPolicy, self).__init__(action_dim, hidden, discrete, **kwargs)

        self.state_dim = state_dim
        self.hidden = hidden

        if self.sac:
            self.fc_mean = nn.Linear(self.hidden[-1], self.action_dim)
            self.fc_std = nn.Linear(self.hidden[-1], self.action_dim)

        self.model = mlp([state_dim] + list(hidden) + [action_dim], sac=self.sac)


class CNNPolicy(BasePolicy):
    """
    CNN Policy

    :param action_dim: Action dimensions of the environment
    :param framestack: Number of previous frames to stack together
    :param hidden: Sizes of hidden layers
    :type action_dim: int
    :type framestack: int
    :type hidden: tuple or list
    """
    def __init__(
        self,
        action_dim: int,
        framestack: int = 4,
        fc_layers: Tuple = (256,),
        discrete: bool = True,
        **kwargs
    ):
        super(CNNPolicy, self).__init__(action_dim, fc_layers, discrete, **kwargs)

        self.action_dim = action_dim

        self.conv, output_size = cnn((framestack, 16, 32))

        self.fc = mlp([output_size] + list(fc_layers) + [action_dim])

    def forward(self, state):
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        state = self.fc(state)
        return state


policy_registry = {"mlp": MlpPolicy, "cnn": CNNPolicy}


def get_policy_from_name(name_: str):
    """
    Returns policy given the name of the policy

    :param name_: Name of the policy needed
    :type name_: str
    :returns: Policy Function to be used
    """
    if name_ in policy_registry:
        return policy_registry[name_]
    raise NotImplementedError
