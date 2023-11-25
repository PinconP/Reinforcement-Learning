import unittest
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os


class TestDQN(unittest.TestCase):

    def test_environment_creation(self):
        """Test if the 'LunarLander-v2' environment is created."""
        env = gym.make("LunarLander-v2")
        self.assertIsNotNone(
            env, "Failed to create LunarLander-v2 environment")

    def test_model_definition(self):
        """Test if the DQN model is defined with the correct parameters."""
        env = gym.make("LunarLander-v2")
        model = DQN(policy='MlpPolicy', env=env, verbose=0)
        self.assertIsNotNone(model, "Failed to define DQN model")

    def test_model_learning(self):
        """Test if the model can start learning."""
        env = gym.make("LunarLander-v2")
        model = DQN(policy='MlpPolicy', env=env, verbose=0)
        try:
            model.learn(total_timesteps=10)
        except Exception as e:
            self.fail(f"Model learning failed with an exception: {e}")

    def test_model_save_and_load(self):
        """Test if the model can be saved and loaded."""
        env = gym.make("LunarLander-v2")
        model = DQN(policy='MlpPolicy', env=env, verbose=0)
        model_name = "test_dqn"
        model.save(model_name)
        loaded_model = DQN.load(model_name)
        self.assertIsNotNone(loaded_model, "Failed to load the saved model")

        if os.path.exists(model_name + ".zip"):
            os.remove(model_name + ".zip")

    def test_model_evaluation(self):
        """Test if the model can be evaluated."""
        env = gym.make("LunarLander-v2")
        model = DQN(policy='MlpPolicy', env=env, verbose=0)
        eval_env = Monitor(gym.make("LunarLander-v2"))
        try:
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True)
        except Exception as e:
            self.fail(f"Model evaluation failed with an exception: {e}")


if __name__ == '__main__':
    unittest.main()
