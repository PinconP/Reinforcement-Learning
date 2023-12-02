import unittest
import gym
import numpy as np
from tqdm import tqdm
import random
from src.q_learning.q_learning import FrozenLakeAgent

env_id = "FrozenLake-v1"


class TestFrozenLakeAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple, deterministic environment for testing
        cls.env = gym.make(env_id, is_slippery=False)
        cls.agent = FrozenLakeAgent(cls.env)

    def test_init(self):
        self.assertEqual(
            self.agent.q_table.shape,
            (self.env.observation_space.n, self.env.action_space.n),
        )
        self.assertTrue(np.all(self.agent.q_table == 0))

    def test_train(self):
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

        # Check if the Q-table has changed from its initial state
        self.assertFalse(np.array_equal(initial_q_table, training_agent.q_table))

    def test_evaluate(self):
        mean_reward, _ = self.agent.evaluate(10, 100)
        self.assertIsInstance(mean_reward, float)

    def test_epsilon_greedy_action(self):
        high_epsilon_action_counts = np.zeros(self.env.action_space.n)
        low_epsilon_action_counts = np.zeros(self.env.action_space.n)
        for _ in range(1000):
            high_epsilon_action = self.agent.epsilon_greedy_action(0, 0.9)
            low_epsilon_action = self.agent.epsilon_greedy_action(0, 0.1)
            high_epsilon_action_counts[high_epsilon_action] += 1
            low_epsilon_action_counts[low_epsilon_action] += 1
        self.assertTrue(np.any(high_epsilon_action_counts > 250))
        self.assertTrue(
            np.argmax(low_epsilon_action_counts) == np.argmax(self.agent.q_table[0])
        )

    def test_update_q_table(self):
        state = 0
        action = 1
        reward = 1
        new_state = 1
        self.agent.update_q_table(state, action, reward, new_state)
        self.assertNotEqual(self.agent.q_table[state, action], 0)

    def test_calculate_epsilon(self):
        epsilon = self.agent.calculate_epsilon(5, 0.01, 1, 0.1)
        self.assertTrue(0.01 <= epsilon <= 1)


if __name__ == "__main__":
    unittest.main()
