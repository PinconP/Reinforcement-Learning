# Importing the gymnasium library to access reinforcement learning environments
import gymnasium as gym

# Importing the DQN algorithm from Stable Baselines3
from stable_baselines3 import DQN
# Utility for creating vectorized environments
from stable_baselines3.common.env_util import make_vec_env
# Function to evaluate the policy performance
from stable_baselines3.common.evaluation import evaluate_policy
# Utility to monitor and log environment data
from stable_baselines3.common.monitor import Monitor
# Callback for evaluating the model during training
from stable_baselines3.common.callbacks import EvalCallback

# Name of the environment to be used
env_name = "LunarLander-v2"

# Create environment using the specified name
env = gym.make(env_name)

# Callback for evaluating the agent periodically
# Saves the best model and monitors performance during training
eval_callback = EvalCallback(
    eval_env=env,  # Single environment for evaluation
    best_model_save_path="./models/dqn",
    eval_freq=100,  # Evaluation frequency
    n_eval_episodes=5,  # Number of evaluation episodes
    verbose=1,
)

# Define the DQN model with specific parameters
model = DQN(
    policy='MlpPolicy',  # Using a Multi-layer Perceptron policy
    env=env,  # The training environment
    learning_rate=0.06771124091067092,  # Learning rate
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
    # Train for a total of 1,000,000 timesteps
    model.learn(total_timesteps=1000000, callback=eval_callback)
except KeyboardInterrupt:
    pass  # Allows the training to be stopped manually
