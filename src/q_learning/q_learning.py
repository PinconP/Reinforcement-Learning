import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm


class FrozenLakeAgent:
    def __init__(self, env, learning_rate=0.7, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def train(self, n_episodes, max_epsilon, min_epsilon, decay_rate, max_steps):
        for episode in tqdm(range(n_episodes), desc="Training"):
            epsilon = self.calculate_epsilon(
                episode, min_epsilon, max_epsilon, decay_rate
            )
            state, _ = self.env.reset()
            for step in range(max_steps):
                action = self.epsilon_greedy_action(state, epsilon)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                if terminated or truncated:
                    break
                state = new_state

    def evaluate(self, n_episodes, max_steps, seed=None):
        total_rewards = []
        for episode in tqdm(range(n_episodes), desc="Evaluating"):
            state, _ = self.env.reset(seed=seed[episode] if seed else None)
            total_reward = 0
            for step in range(max_steps):
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(total_reward)
        return np.mean(total_rewards), np.std(total_rewards)

    def epsilon_greedy_action(self, state, epsilon):
        if random.uniform(0, 1) > epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()

    def update_q_table(self, state, action, reward, new_state):
        best_next_action = np.argmax(self.q_table[new_state])
        self.q_table[state, action] += self.learning_rate * (
            reward
            + self.gamma * self.q_table[new_state, best_next_action]
            - self.q_table[state, action]
        )

    @staticmethod
    def calculate_epsilon(episode, min_epsilon, max_epsilon, decay_rate):
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


def create_environment(env_id="FrozenLake-v1", map_name="4x4", is_slippery=False):
    return gym.make(
        env_id, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array"
    )


def print_environment_info(env):
    print(
        f"_____OBSERVATION SPACE_____ \nObservation Space: {env.observation_space}\nSample Observation: {env.observation_space.sample()}"
    )
    print(
        f"\n_____ACTION SPACE_____ \nAction Space Shape: {env.action_space.n}\nAction Space Sample: {env.action_space.sample()}"
    )


# Main execution
if __name__ == "__main__":
    env = create_environment()
    print_environment_info(env)

    agent = FrozenLakeAgent(env)

    # Training parameters
    n_training_episodes = 10000
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005
    max_steps = 99

    agent.train(n_training_episodes, max_epsilon, min_epsilon, decay_rate, max_steps)

    # Evaluation parameters
    n_eval_episodes = 100
    mean_reward, std_reward = agent.evaluate(n_eval_episodes, max_steps)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
