import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cv2  # Pour l'affichage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Name of the environment to be used
env_name = "CartPole-v1"

# Charger l'environnement vectorisé
env = make_vec_env(env_name, n_envs=1)

# Charger le modèle entraîné
model_name = "models/ppo/best_model"
model = PPO.load(model_name)


eval_env = Monitor(gym.make(env_name))
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


# Exécuter l'environnement pour visualiser le modèle
num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render(mode="human")
env.close()
