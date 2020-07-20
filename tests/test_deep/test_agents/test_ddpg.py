import shutil

from genrl import DDPG
from genrl.deep.common import NormalActionNoise, OffPolicyTrainer
from genrl.environments import VectorEnv


def test_ddpg():
    env = VectorEnv("Pendulum-v0", 2)
    algo = DDPG("mlp", env, noise=NormalActionNoise, layers=[1, 1], replay_size=200)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=500
    )
    trainer.train()
    shutil.rmtree("./logs")
