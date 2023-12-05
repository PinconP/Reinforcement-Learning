# Importing the gymnasium library to access reinforcement learning environments
import gymnasium as gym

# Importing the A2C algorithm from Stable Baselines3
from stable_baselines3 import A2C

# Utility for creating vectorized environments
from stable_baselines3.common.env_util import make_vec_env

# Function to evaluate the policy performance
from stable_baselines3.common.evaluation import evaluate_policy

# Utility to monitor and log environment data
from stable_baselines3.common.monitor import Monitor

# Callback for evaluating the model during training
from stable_baselines3.common.callbacks import EvalCallback

# Name of the environment to be used
env_name = "BipedalWalkerHardcore-v3"

# Create environment using the specified name
env = make_vec_env(env_name, n_envs=16)

# Create multiple instances of the environment for evaluation
eval_envs = make_vec_env(env_name, n_envs=5)

# Callback for evaluating the agent periodically
# Saves the best model and monitors performance during training
eval_callback = EvalCallback(
    # Evaluation environments: Environments used for evaluating the agent.
    eval_envs,
    # In this case, it's multiple instances of the environment "env_name".
    best_model_save_path="./models/a2c",
    # Best model save path: Directory where the best model will be saved.
    # "Best" is determined based on the performance in the evaluation environments.
    eval_freq=50000,
    # Evaluation frequency: Determines how often the evaluation should be done.
    # In this case, it's set to 50000, meaning evaluation at the end of each 50000 training step.
    n_eval_episodes=10,
    # Number of evaluation episodes: Number of episodes to run for each evaluation.
    # Here, it means the agent will be evaluated over 10 episodes each time.
    verbose=1,
    # Verbose: Verbosity level (0: no output, 1: info, 2: debug).
    # Note: There are other parameters available in EvalCallback not used that could be interesting:
    # render: If True, the environments will be rendered during evaluation.
)


# Define the A2C model with specific parameters
model = A2C(
    policy="MlpPolicy",  # Using a Multi-layer Perceptron policy
    env=env,  # The training environment
    buffer_size=100000,  # Size of the replay buffer
    learning_starts=1000,  # Number of steps before learning starts
    batch_size=256,  # Size of the batch for learning the policy
    # The soft update coefficient (tau) for updating the target network
    tau=0.6603913103837233,
    gamma=0.935566138348333,  # Discount factor
    train_freq=5,  # Update the model every 5 steps
    gradient_steps=5,  # How many gradient steps to do after each rollout
    optimize_memory_usage=False,  # Optimize memory usage
    verbose=0,  # Verbose mode
)

# Start training the agent
try:
    # Train for a total of 5 million timesteps
    model.learn(total_timesteps=int(5e6), callback=eval_callback)
except KeyboardInterrupt:
    pass  # Allows the training to be stopped manually
