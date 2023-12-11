import unittest
import gym
import numpy as np
from tqdm import tqdm
import random
from src.q_learning.q_learning import FrozenLakeAgent, create_environment

env_id = "FrozenLake-v1"


class TestFrozenLakeAgent(unittest.TestCase):
    def setUp(self):
        self.env = create_environment()
        self.agent = FrozenLakeAgent(self.env)

    def test_environment_creation(self):
        self.assertIsNotNone(self.env)
        self.assertGreater(self.env.observation_space.n, 0)
        self.assertGreater(self.env.action_space.n, 0)

    def test_agent_initialization(self):
        self.assertEqual(
            self.agent.q_table.shape,
            (self.env.observation_space.n, self.env.action_space.n),
        )
        self.assertGreater(self.agent.learning_rate, 0)
        self.assertGreater(self.agent.gamma, 0)

    def test_training_process(self):
        # Create a separate instance of the environment for training
        training_env = gym.make(env_id, is_slippery=False)
        training_agent = FrozenLakeAgent(training_env)

        initial_q_table = np.copy(training_agent.q_table)

        # Train the agent
        training_agent.train(
            n_episodes=1000,
            max_epsilon=1.0,
            min_epsilon=0.01,
            decay_rate=0.005,
            max_steps=100,
        )

    def test_epsilon_greedy_action(self):
        state = 0
        action_high_epsilon = self.agent.epsilon_greedy_action(state, 0.9)
        action_low_epsilon = self.agent.epsilon_greedy_action(state, 0.1)
        self.assertIn(action_high_epsilon, range(self.env.action_space.n))
        self.assertIn(action_low_epsilon, range(self.env.action_space.n))

    def test_q_table_update(self):
        state, action, reward, new_state = 0, 1, 1, 1
        original_value = self.agent.q_table[state, action]
        self.agent.update_q_table(state, action, reward, new_state)
        self.assertNotEqual(self.agent.q_table[state, action], original_value)

    def test_epsilon_decay(self):
        epsilon_initial = FrozenLakeAgent.calculate_epsilon(0, 0.05, 1.0, 0.01)
        epsilon_final = FrozenLakeAgent.calculate_epsilon(1000, 0.05, 1.0, 0.01)
        self.assertGreater(epsilon_initial, epsilon_final)
        self.assertGreaterEqual(epsilon_final, 0.05)

    def test_evaluation_function(self):
        self.agent.train(10, 1.0, 0.05, 0.001, 100)
        mean_reward, std_reward = self.agent.evaluate(10, 100)
        self.assertGreaterEqual(mean_reward, 0)
        self.assertGreaterEqual(std_reward, 0)


if __name__ == "__main__":
    unittest.main()
