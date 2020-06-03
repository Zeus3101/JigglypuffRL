from genrl.environments import AtariEnv, GymEnv     # noqa
from genrl.deep.common import OnPolicyTrainer       # noqa
from genrl import PPO1, SAC                         # noqa


if __name__ == "__main__":
    env = AtariEnv(
        "PongNoFrameskip-v0", lz4_compress=True,
    )
    agent = PPO1("cnn", env)
    # env = GymEnv("CartPole-v0")
    # agent = PPO1("mlp", env)
    # trainer = OnPolicyTrainer(
    #     agent, epochs=5, save_interval=5, log_interval=1,
    #     ckpt_log_name="checkpoints"
    # )
    # trainer.train()
    agent.learn()
