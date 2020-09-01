import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from genrl.agents import A2C
from genrl.core.rollout_storage import RolloutBuffer
from genrl.utils.utils import get_env_properties

from ..utils import CentralizedActorCritic


class MAA2C(A2C):
    def __init__(self, *args, **kwargs):
        super(MAA2C, self).__init__(*args, **kwargs)
        self.num_agents = self.env.n

        if self.create_model:
            self._create_model()

    def _create_model(self, load_model):
        state_dim, action_dim, discrete, _ = get_env_properties(self.env, self.network)

        self.ac = CentralizedActorCritic(state_dim, action_dim).to(self.device)
        if load_model is not None:
            self.ac.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

    def select_action(self, state, deterministic):
        state = torch.as_tensor(state).float().to(self.device)

        actions, log_probs, values = [], [], []
        for i in range(self.num_agents):
            action, dist = self.ac.get_action(state[i], deterministic=deterministic)
            value = self.ac.get_value(state[i])

            actions.append(action.detach().cpu().numpy())
            log_probs.append(dist.log_prob(action).cpu())
            values.append(value)

        return action.detach().cpu().numpy(), value, dist.log_prob(action).cpu()

    # def collect_rollouts(self, states):
    #     current_states = states

    #     for step in range(self.steps_per_episode):

    #         actions = self.get_actions(current_states)
    #         next_states, rewards, dones, info = self.env.step(actions)
    #         self.epoch_reward += np.sum(rewards)

    #         if all(dones) or step == self.steps_per_episode - 1:

    #             dones = [1 for _ in range(self.num_agents)]
    #             self.rollout.push(
    #                 (
    #                     torch.FloatTensor(states),
    #                     torch.LongTensor(actions),
    #                     torch.FloatTensor(rewards),
    #                     torch.FloatTensor(next_states),
    #                     torch.LongTensor(dones),
    #                 )
    #             )
    #             print("REWARD: {} \n".format(np.round(self.epoch_reward, decimals=4)))
    #             print("*" * 100)
    #             self.final_step = step
    #             break
    #         else:
    #             dones = [0 for _ in range(self.num_agents)]
    #             self.rollout.push(
    #                 (
    #                     torch.FloatTensor(states),
    #                     torch.LongTensor(actions),
    #                     torch.FloatTensor(rewards),
    #                     torch.FloatTensor(next_states),
    #                     torch.LongTensor(dones),
    #                 )
    #             )
    #             current_states = next_states
    #             self.final_step = step

    #     self.states = torch.FloatTensor([sars[0] for sars in self.rollout]).to(
    #         self.device
    #     )
    #     self.next_states = torch.FloatTensor([sars[1] for sars in self.rollout]).to(
    #         self.device
    #     )
    #     self.actions = torch.LongTensor([sars[2] for sars in self.rollout]).to(
    #         self.device
    #     )
    #     self.rewards = torch.FloatTensor([sars[3] for sars in self.rollout]).to(
    #         self.device
    #     )
    #     self.dones = torch.LongTensor([sars[4] for sars in self.rollout])

    #     self.logits, self.values = self.ac.forward(self.states)

    #     return self.values, self.dones

    # def get_traj_loss(self, curr_Q, done):
    #     discounted_rewards = np.asarray(
    #         [
    #             [
    #                 torch.sum(
    #                     torch.FloatTensor(
    #                         [
    #                             self.gamma ** i
    #                             for i in range(self.rewards[k][j:].size(0))
    #                         ]
    #                     )
    #                     * self.rewards[k][j:]
    #                 )
    #                 for j in range(self.rewards.size(0))
    #             ]
    #             for k in range(self.num_agents)
    #         ]
    #     )
    #     discounted_rewards = np.transpose(discounted_rewards)
    #     value_targets = self.rewards + torch.FloatTensor(discounted_rewards).to(
    #         self.device
    #     )
    #     value_targets = value_targets.unsqueeze(dim=-1)
    #     self.value_loss = F.smooth_l1_loss(curr_Q, value_targets)

    #     dists = F.softmax(self.logits, dim=-1)
    #     probs = Categorical(dists)

    #     self.entropy = -torch.mean(
    #         torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=-1)
    #     )

    #     advantage = value_targets - curr_Q
    #     self.policy_loss = (
    #         -probs.log_prob(self.actions).unsqueeze(dim=-1) * advantage.detach()
    #     )
    #     self.policy_loss = self.policy_loss.mean()

    #     self.total_loss = self.policy_loss + self.value_loss - self.entropy_coeff * self.entropy

    # def update_params(self):
    #     self.optimizer.zero_grad()
    #     self.total_loss.backward(retain_graph=False)
    #     self.grad_norm = torch.nn.utils.clip_grad_norm_(
    #         self.ac.parameters(), 0.5
    #     )
    #     self.optimizer.step()

    # def get_logging_params(self):
    #     logging_params = {
    #         "Loss/Entropy loss": self.entropy.item(),
    #         "Loss/Value Loss": self.value_loss.item(),
    #         "Loss/Policy Loss": self.policy_loss,
    #         "Loss/Total Loss": self.total_loss,
    #         "Gradient Normalization/Grad Norm": self.grad_norm,
    #         "Reward Incurred/Length of the episode": self.final_step,
    #         "Reward Incurred/Reward": self.epoch_reward,
    #     }
    #     return logging_params

    # def get_hyperparams(self):
    #     hyperparams = {
    #         "gamma": self.gamma,
    #         "entropy_coeff": self.entropy_coeff,
    #         "lr_actor": self.lr,
    #         "lr_critic": self.lr,
    #         "actorcritic_weights": self.ac.state_dict(),
    #     }

    #     return hyperparams
