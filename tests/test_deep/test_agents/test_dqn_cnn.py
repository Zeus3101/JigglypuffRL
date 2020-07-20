import shutil

from genrl.deep.agents import (
    DQN,
    CategoricalDQN,
    DoubleDQN,
    DuelingDQN,
    NoisyDQN,
    PrioritizedReplayDQN,
)
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


class TestDQNCNN:
    def test_vanilla_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_double_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DoubleDQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_dueling_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DuelingDQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_prioritized_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = PrioritizedReplayDQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_noisy_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = NoisyDQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_categorical_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = CategoricalDQN("cnn", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", steps_per_epoch=200, epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")
