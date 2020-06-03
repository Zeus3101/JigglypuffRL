import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
import gym
import time

from ...common import (
    get_model,
    save_params,
    load_params,
    get_env_properties,
    set_seeds,
    venv,
)

from typing import Union, Any, Optional, Tuple, Dict


class PPO1:
    """
    Proximal Policy Optimization algorithm (Clipped policy).

    Paper: https://arxiv.org/abs/1707.06347

    :param network_type: (str) The deep neural network layer types ['mlp']
    :param env: (Gym environment) The environment to learn from
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param gamma: (float) discount factor
    :param clip_param: (float) clipping parameter epsilon
    :param actor_batchsize: (int) trajectories per optimizer epoch
    :param epochs: (int) the optimizer's number of epochs
    :param lr_policy: (float) policy network learning rate
    :param lr_value: (float) value network learning rate
    :param policy_copy_interval: (int) number of optimizer before copying
        params from new policy to old policy
    :param save_interval: (int) Number of episodes between saves of models
    :param tensorboard_log: (str) the log location for tensorboard (if None,
        no logging)
    :param seed (int): seed for torch and gym
    :param device (str): device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :param run_num: (boolean) if model has already been trained
    :param save_model: (string) directory the user wants to save models to
    :param load_model: model loading path
    :type load_model: string
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, venv],
        timesteps_per_actorbatch: int = 256,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        actor_batch_size: int = 64,
        epochs: int = 1000,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        layers: Tuple = (64, 64),
        policy_copy_interval: int = 20,
        tensorboard_log: str = None,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
    ):
        self.network_type = network_type
        self.env = env
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.gamma = gamma
        self.clip_param = clip_param
        self.actor_batch_size = actor_batch_size
        self.epochs = epochs
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.layers = layers
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.policy_copy_interval = policy_copy_interval
        self.save_interval = save_interval
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save = save_params
        self.load = load_params

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # init writer if tensorboard
        self.writer = None
        if self.tensorboard_log is not None:  # pragma: no cover
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)
        self.create_model()

    def create_model(self) -> None:
        # Instantiate networks and optimizers
        state_dim, action_dim, discrete, action_lim = get_env_properties(self.env)

        if self.network_type == "mlp":
            self.policy_new, self.policy_old = (
                get_model("p", self.network_type)(
                    state_dim, action_dim, self.layers,
                    discrete=discrete, action_lim=action_lim
                ),
                get_model("p", self.network_type)(
                    state_dim, action_dim, self.layers,
                    discrete=discrete, action_lim=action_lim
                ),
            )
            self.policy_new = self.policy_new.to(self.device)
            self.policy_old = self.policy_old.to(self.device)

            self.value_fn = get_model("v", self.network_type)(
                state_dim, action_dim, "V"
            ).to(self.device)

        elif self.network_type == "cnn":
            self.framestack = self.env.framestack

            self.policy_new, self.policy_old = (
                get_model("p", self.network_type)(
                    action_dim, self.framestack, self.layers,
                    discrete=discrete, action_lim=action_lim
                ),
                get_model("p", self.network_type)(
                    action_dim, self.framestack, self.layers,
                    discrete=discrete, action_lim=action_lim
                ),
            )
            self.policy_new = self.policy_new.to(self.device)
            self.policy_old = self.policy_old.to(self.device)

            self.value_fn = get_model("v", self.network_type)(
                action_dim, self.framestack, "V"
            ).to(self.device)

        # load paramaters if already trained
        if self.load_model is not None:
            self.load(self)
            self.policy_new.load_state_dict(self.checkpoint["policy_weights"])
            self.value_fn.load_state_dict(self.checkpoint["value_weights"])
            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.policy_old.load_state_dict(self.policy_new.state_dict())

        self.optimizer_policy = opt.Adam(
            self.policy_new.parameters(), lr=self.lr_policy
        )
        self.optimizer_value = opt.Adam(self.value_fn.parameters(), lr=self.lr_value)

        self.traj_reward = []
        self.policy_old.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.policy_new.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.value_fn.value_hist = Variable(torch.Tensor()).to(self.device)

        self.policy_new.loss_hist = Variable(torch.Tensor()).to(self.device)
        self.value_fn.loss_hist = Variable(torch.Tensor()).to(self.device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on policy_old output
        action, c_old = self.policy_old.get_action(
            Variable(state), deterministic=False
        )
        _, c_new = self.policy_new.get_action(
            Variable(state), deterministic=False
        )
        val = self.value_fn.get_value(Variable(state))

        # store policy probs and value function for current traj
        self.policy_old.policy_hist = torch.cat(
            [
                self.policy_old.policy_hist,
                c_old.log_prob(action).exp().prod().unsqueeze(0),
            ]
        )

        self.policy_new.policy_hist = torch.cat(
            [
                self.policy_new.policy_hist,
                c_new.log_prob(action).exp().prod().unsqueeze(0),
            ]
        )

        self.value_fn.value_hist = torch.cat(
            [self.value_fn.value_hist, val.unsqueeze(0)]
        )

        action = action.detach().cpu().numpy()
        return action

    # get clipped loss for single trajectory (episode)
    def get_traj_loss(self) -> None:
        discounted_reward = 0
        returns = []

        # calculate discounted return
        for reward in self.traj_reward[::-1]:
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # advantage estimation
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = Variable(returns) - Variable(self.value_fn.value_hist)

        # compute policy and value loss
        ratio = torch.div(
            self.policy_new.policy_hist, self.policy_old.policy_hist
        )
        clipping = (
            torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            .mul(advantages)
            .to(self.device)
        )

        loss_policy = (
            torch.mean(torch.min(torch.mul(ratio, advantages), clipping))
            .mul(-1)
            .unsqueeze(0)
        )
        loss_value = nn.MSELoss()(
            self.value_fn.value_hist, Variable(returns)
        ).unsqueeze(0)

        # store traj loss values in epoch loss tensors
        self.policy_new.loss_hist = torch.cat([
            self.policy_new.loss_hist, loss_policy
        ])
        self.value_fn.loss_hist = torch.cat([
            self.value_fn.loss_hist, loss_value
        ])

        # clear traj history
        self.traj_reward = []
        self.policy_old.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.policy_new.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.value_fn.value_hist = Variable(torch.Tensor()).to(self.device)

    def update(self, episode: int, copy_policy: bool = True) -> None:
        # mean of all traj losses in single epoch
        start = time.perf_counter()
        loss_policy = torch.mean(self.policy_new.loss_hist)
        loss_value = torch.mean(self.value_fn.loss_hist)
        end = time.perf_counter()
        print("Mean: {}".format(end-start))

        # tensorboard book-keeping
        if self.tensorboard_log:
            self.writer.add_scalar("loss/policy", loss_policy, episode)
            self.writer.add_scalar("loss/value", loss_value, episode)

        # take gradient step
        start = time.perf_counter()
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()
        end = time.perf_counter()
        print("Policy backward: {}".format(end-start))

        start = time.perf_counter()
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()
        end = time.perf_counter()
        print("Value backward: {}".format(end-start))

        # clear loss history for epoch
        start = time.time()
        self.policy_new.loss_hist = Variable(torch.Tensor()).to(self.device)
        self.value_fn.loss_hist = Variable(torch.Tensor()).to(self.device)
        end = time.time()
        print("Clear: {}".format(end-start))

        if copy_policy:
            self.policy_old.load_state_dict(self.policy_new.state_dict())

    def learn(self) -> None:  # pragma: no cover
        # training loop
        for episode in range(self.epochs):
            epoch_reward = 0

            select_action_time = 0
            step_time = 0
            get_traj_loss_time = 0
            update_time = 0
            for i in range(self.actor_batch_size):
                state = self.env.reset()
                done = False
                for t in range(self.timesteps_per_actorbatch):
                    start = time.perf_counter()
                    action = self.select_action(state)
                    end = time.perf_counter()
                    select_action_time += end-start
                    # print("Select Action: {}".format(end-start))

                    start = time.perf_counter()
                    state, reward, done, _ = self.env.step(action)
                    end = time.perf_counter()
                    step_time += end-start
                    # if end-start > 0.0:
                    #     print("Step: {}, Action: {}".format(end-start, action))

                    if self.render:
                        self.env.render()

                    self.traj_reward.append(reward)

                    if done:
                        break

                epoch_reward += (
                    np.sum(self.traj_reward) / self.actor_batch_size
                )
                start = time.perf_counter()
                self.get_traj_loss()
                end = time.perf_counter()
                get_traj_loss_time += end-start
                # print("Get traj loss: {}".format(end-start))

            start = time.perf_counter()
            self.update(episode)
            end = time.perf_counter()
            update_time += end-start

            if episode % 1 == 0:
                print("\n\nEpisode: {}, reward: {}\n".format(episode, epoch_reward))
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", epoch_reward, episode)
                print("Update: {}".format(update_time))
                print("Get Traj Loss: {}".format(get_traj_loss_time))
                print("Step: {}".format(step_time))
                print("Select Action: {}".format(select_action_time))

            if episode % self.policy_copy_interval == 0:
                self.policy_old.load_state_dict(self.policy_new.state_dict())

            if self.save_model is not None:
                if episode % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, episode)
                    print("Saved current model")

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "actor_batch_size": self.actor_batch_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "policy_weights": self.policy_new.state_dict(),
            "value_weights": self.value_fn.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    algo = PPO1("mlp", env)
    algo.learn()
