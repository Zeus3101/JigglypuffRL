from genrl.environments import VectorEnv
from genrl.multi import MAA2C
from genrl.trainers import OnPolicyTrainer

env = VectorEnv("simple_spread", env_type="multi")
agent = MAA2C("mlpshared", env)
trainer = OnPolicyTrainer(
    agent, env, log_mode=["stdout", "tensorboard"], logdir="./runs"
)
trainer.train()
trainer.evaluate()
