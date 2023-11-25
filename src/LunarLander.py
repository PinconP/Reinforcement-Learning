import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

env_name = "LunarLander-v2"

# Create environment
env = gym.make(env_name)

# Create the evaluation envs
eval_envs = make_vec_env(env_name, n_envs=5)


# Create evaluation callback to save best model
# and monitor agent performance
eval_callback = EvalCallback(
    eval_envs,
    best_model_save_path="./models/ppo-LunarLander",
    eval_freq=1,
    n_eval_episodes=10,
)

# Model definition
model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1)

# Train the agent (you can kill it before using ctrl+c)
try:
    model.learn(total_timesteps=int(5e6), callback=eval_callback)
except KeyboardInterrupt:
    pass
