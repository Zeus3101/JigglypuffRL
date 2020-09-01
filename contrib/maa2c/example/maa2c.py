import numpy as np

from genrl.environments import VectorEnv
from genrl.multi.maa2c import MAA2C
from genrl.trainers import OnPolicyTrainer


def train():
    env = VectorEnv("simple_spread", env_type="multi")
    agent = MAA2C("mlpshared", env)
    trainer = OnPolicyTrainer(
        agent, env, log_mode=["stdout", "tensorboard"], logdir="./runs"
    )
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    train()
