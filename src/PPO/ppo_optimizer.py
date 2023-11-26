import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import optuna

env_name = "LunarLander-v2"  # Name of the environment to be used
env = make_vec_env(env_name, n_envs=16)
eval_envs = make_vec_env(env_name, n_envs=5)


def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_float('ent_coef', 0.0001, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)

    # Create a PPO model with these hyperparameters
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda,
                ent_coef=ent_coef, n_epochs=n_epochs, verbose=0)

    # Train the model for a fixed number of timesteps
    model.learn(total_timesteps=100000)  # Smaller number for trial runs

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_envs, n_eval_episodes=10)

    return mean_reward


# Optimization study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)  # Define the number of trials

# Best hyperparameters
print("Best hyperparameters:", study.best_params)
