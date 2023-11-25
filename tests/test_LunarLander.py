import unittest
# Importation de la bibliothèque gymnasium pour les environnements de RL
import gymnasium as gym
from stable_baselines3 import PPO  # Importation de l'algorithme PPO
# Outil d'évaluation de politiques
from stable_baselines3.common.evaluation import evaluate_policy
# Pour enregistrer les données de l'environnement
from stable_baselines3.common.monitor import Monitor
import os  # Pour les opérations liées au système de fichiers


class TestReinforcementLearning(unittest.TestCase):
    # Classe de test pour les fonctionnalités de RL

    def test_environment_creation(self):
        """Test si l'environnement LunarLander-v2 est créé."""
        env = gym.make('LunarLander-v2')  # Création de l'environnement
        self.assertIsNotNone(
            env, "Failed to create LunarLander-v2 environment")

    def test_model_definition(self):
        """Test si le modèle PPO est défini avec les bons paramètres."""
        env = gym.make('LunarLander-v2')
        # Définition du modèle PPO
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
        self.assertIsNotNone(model, "Failed to define PPO model")

    def test_model_learning(self):
        """Test si le modèle peut commencer à apprendre."""
        env = gym.make('LunarLander-v2')
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
        try:
            # Tentative d'apprentissage du modèle
            model.learn(total_timesteps=10)
        except Exception as e:
            self.fail(f"Model learning failed with an exception: {e}")

    def test_model_save_and_load(self):
        """Test si le modèle peut être sauvegardé et chargé."""
        env = gym.make('LunarLander-v2')
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
        model_name = "test_ppo_LunarLander"
        model.save(model_name)  # Sauvegarde du modèle
        loaded_model = PPO.load(model_name)  # Chargement du modèle
        self.assertIsNotNone(loaded_model, "Failed to load the saved model")

        # Suppression du fichier modèle après le test
        if os.path.exists(model_name + ".zip"):
            os.remove(model_name + ".zip")

    def test_model_evaluation(self):
        """Test si le modèle peut être évalué."""
        env = gym.make('LunarLander-v2')
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
        # Création d'un environnement pour l'évaluation
        eval_env = Monitor(gym.make("LunarLander-v2"))
        try:
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True)  # Évaluation du modèle
        except Exception as e:
            self.fail(f"Model evaluation failed with an exception: {e}")


if __name__ == '__main__':
    unittest.main()
