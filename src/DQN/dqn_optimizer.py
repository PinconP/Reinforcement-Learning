import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import optuna

# Name of the environment to be used
env_name = "LunarLander-v2"
env = gym.make(env_name)
eval_envs = make_vec_env(env_name, n_envs=5)


def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    buffer_size = trial.suggest_categorical(
        'buffer_size', [10000, 50000, 100000])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    train_freq = trial.suggest_int('train_freq', 1, 10)
    gradient_steps = trial.suggest_int('gradient_steps', 1, 10)
    tau = trial.suggest_float('tau', 0.001, 1.0)

    # Create a DQN model with these hyperparameters
    model = DQN('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size,
                batch_size=batch_size, gamma=gamma, train_freq=train_freq,
                gradient_steps=gradient_steps, tau=tau, verbose=0)

    # Train the model for a fixed number of timesteps
    model.learn(total_timesteps=20000)  # Smaller number for trial runs

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_envs, n_eval_episodes=10)

    return mean_reward


# Optimization study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Define the number of trials

# Best hyperparameters
print("Best hyperparameters:", study.best_params)
