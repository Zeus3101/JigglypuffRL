import collections
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as opt

from genrl.agents import OffPolicyAgentAC
from genrl.core import ActionNoise
from genrl.core.models import VAE
from genrl.utils.utils import get_env_properties, get_model, safe_mean


class BCQ(OffPolicyAgentAC):
    """Batch Constrained Q-Learning

    Paper: https://arxiv.org/abs/1812.02900

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        policy_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the policy
        value_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the critics
        shared_layers (:obj:`tuple` of :obj:`int`): Sizes of shared layers in Actor Critic if using
        vae_layers (:obj:`tuple` of :obj:`int`): Sizes of hidden layers in the VAE
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        lr_vae (float): Learning rate for the VAE
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        noise (:obj:`ActionNoise`): Action Noise function added to aid in exploration
        noise_std (float): Standard deviation of the action noise distribution
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        noise: ActionNoise = None,
        noise_std: float = 0.2,
        vae_layers: Tuple = (32, 32),
        lr_vae: float = 0.001,
        **kwargs
    ):
        super(BCQ, self).__init__(*args, **kwargs)
        self.noise = noise
        self.noise_std = noise_std
        self.vae_layers = vae_layers
        self.lr_vae = lr_vae
        self.doublecritic = True

        self._create_model()
        self.empty_logs()

    def _create_model(self) -> None:
        """Function to initialize the BCQ Model

        This will create the BCQ Q-networks and the VAE
        """
        state_dim, action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network
        )
        if discrete:
            raise Exception(
                "Only continuous Environments are supported for the original BCQ. For discrete BCQ, use DiscreteBCQ instead"
            )

        if isinstance(self.network, str):
            arch = self.network + "12"
            if self.shared_layers is not None:
                arch += "s"
            self.ac = get_model("ac", arch)(
                state_dim,
                action_dim,
                shared_layers=self.shared_layers,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_type="Qsa",
                discrete=False,
            )
            self.vae = VAE(
                state_dim, action_dim, self.vae_layers, action_lim, self.device
            )
        else:
            self.ac, self.vae = self.network

        if self.noise is not None:
            self.noise = self.noise(
                torch.zeros(action_dim), self.noise_std * torch.ones(action_dim)
            )

        self.ac_target = deepcopy(self.ac)
        actor_params, critic_params = self.ac.get_params()
        self.optimizer_value = opt.Adam(critic_params, lr=self.lr_value)
        self.optimizer_policy = opt.Adam(actor_params, lr=self.lr_policy)
        self.optimizer_vae = opt.Adam(self.vae.parameters(), lr=self.lr_vae)

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()

            vae_loss = self.get_vae_loss(batch)

            self.optimizer_vae.zero_grad()
            vae_loss.backward()
            self.optimizer_vae.step()

            value_loss = self.get_q_loss(batch)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            policy_loss = self.get_p_loss(batch.states)

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.logs["policy_loss"].append(policy_loss.item())
            self.logs["value_loss"].append(value_loss.item())
            self.logs["vae_loss"].append(vae_loss.item())

            self.update_target_model()

    def get_vae_loss(self, batch: collections.namedtuple) -> None:
        """BCQ Function to calculate the loss of the VAE

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the VAE of the BCQ
        """
        recon, mean, std = self.vae(batch.states, batch.actions)
        recon_loss = F.mse_loss(recon, batch.actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        return vae_loss

    def get_p_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Function to get the Policy loss

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found

        Returns:
            loss (:obj:`torch.Tensor`): Calculated policy loss
        """
        sampled_actions = self.vae.decode(states)
        sampled_action_state = torch.cat([states, sampled_actions], dim=-1)
        perturbed_actions = self.ac.get_action(
            sampled_action_state, deterministic=True
        )[0]
        perturbed_action_state = torch.cat([states, perturbed_actions], dim=-1)

        q_values = self.ac.get_value(perturbed_action_state)
        policy_loss = -torch.mean(q_values)
        return policy_loss

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Get target Q values for the TD3

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the TD3
        """
        next_state_actions = torch.cat(
            [next_states, self.vae.decode(next_states)], dim=-1
        )
        next_target_actions = self.ac_target.get_action(next_state_actions, True)[0]

        next_q_target_values = self.ac_target.get_value(
            torch.cat([next_states, next_target_actions], dim=-1), mode="min"
        )
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target_values

        return target_q_values

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_size": self.replay_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "lr_vae": self.lr_vae,
            "polyak": self.polyak,
            "noise_std": self.noise_std,
        }

        return hyperparams, self.ac.state_dict()

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
            "vae_loss": safe_mean(self.logs["vae_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs"""
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["vae_loss"] = []