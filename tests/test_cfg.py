import unittest
import os
from pathlib import Path

# Ensure src is in python path for testing
import sys
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config import AppConfig, cfg as global_cfg # Import the AppConfig class and the global cfg instance

class TestConfig(unittest.TestCase):

    def test_default_config_loading(self):
        """Test that the global cfg instance loads default.yaml correctly."""
        self.assertIsInstance(global_cfg, AppConfig)
        self.assertEqual(global_cfg.slm.model_name, "Phi-3-Mini-128k-Instruct")
        self.assertEqual(global_cfg.mcts.simulations_per_step, 8)
        self.assertEqual(global_cfg.random_seed, 42)
        self.assertTrue(hasattr(global_cfg, "datasets"))
        self.assertTrue(hasattr(global_cfg.datasets, "prompt_seeds"))
        self.assertEqual(global_cfg.datasets.prompt_seeds.name, "wikitext")
        self.assertTrue(hasattr(global_cfg, "features"))
        self.assertTrue(global_cfg.features.entropy.enabled)
        self.assertEqual(global_cfg.features.z_entropy.window_size, 16)
        self.assertEqual(global_cfg.macro_actions.branch.k, 3)

    def test_config_structure_complete(self):
        """Test that all top-level and some nested keys are present in the loaded config."""
        required_top_level_keys = [
            "slm", "datasets", "features", "macro_actions", "mcts",
            "self_play", "training", "long_run", "kenlm", "openai",
            "random_seed", "log_level"
        ]
        for key in required_top_level_keys:
            self.assertTrue(hasattr(global_cfg, key), msg=f"Top-level key '{key}' missing in config.")

        # Test some nested structures
        self.assertIsNotNone(global_cfg.slm.special_tokens)
        self.assertIsNotNone(global_cfg.datasets.prompt_seeds)
        self.assertIsNotNone(global_cfg.datasets.topic_drift_embeddings)
        self.assertIsNotNone(global_cfg.datasets.qa_eval)
        self.assertIsNotNone(global_cfg.features.context_vec)
        self.assertIsNotNone(global_cfg.macro_actions.branch)
        self.assertIsNotNone(global_cfg.self_play.reward)

    def test_loading_non_existent_file_falls_back_to_defaults(self):
        """Test that AppConfig falls back to defaults if YAML is not found."""
        # Temporarily rename the actual config to simulate its absence
        actual_config_path = Path("configs/default.yaml")
        temp_config_path = Path("configs/default.yaml.temp_test")
        renamed = False
        if actual_config_path.exists():
            os.rename(actual_config_path, temp_config_path)
            renamed = True
        
        try:
            cfg_no_file = AppConfig.from_yaml(actual_config_path) # This should now raise FileNotFoundError internally
            self.assertIsInstance(cfg_no_file, AppConfig)
            # Check a few default values to confirm fallback
            self.assertEqual(cfg_no_file.slm.model_name, "Phi-3-Mini-128k-Instruct") # Default from dataclass
            self.assertEqual(cfg_no_file.random_seed, 42) # Default from dataclass
        finally:
            # Restore the config file if it was renamed
            if renamed:
                os.rename(temp_config_path, actual_config_path)

    def test_types_of_config_values(self):
        """Test that critical config values have the expected types."""
        self.assertIsInstance(global_cfg.slm.context_length, int)
        self.assertIsInstance(global_cfg.slm.special_tokens.assistant_thought_begin, str)
        self.assertIsInstance(global_cfg.datasets.prompt_seeds.min_prompt_words, int)
        self.assertIsInstance(global_cfg.features.z_entropy.window_size, int)
        self.assertIsInstance(global_cfg.features.context_vec.hidden_size, int)
        self.assertIsInstance(global_cfg.macro_actions.temperature_bump.factor, float)
        self.assertIsInstance(global_cfg.mcts.ucb_c, float)
        self.assertIsInstance(global_cfg.training.learning_rate, float)
        self.assertIsInstance(global_cfg.training.entropy_bonus_weight, float)
        self.assertIsInstance(global_cfg.kenlm.auto_download, bool)
        self.assertIsInstance(global_cfg.random_seed, int)
        self.assertIsInstance(global_cfg.log_level, str)

if __name__ == "__main__":
    # Create a dummy configs directory and default.yaml for isolated testing if needed
    # This is more for running this test file directly.
    # In a pytest scenario, the project structure is usually available.
    test_config_dir = Path("configs")
    if not test_config_dir.exists():
        test_config_dir.mkdir(parents=True, exist_ok=True)
        dummy_yaml_content = """
slm:
  model_name: "Phi-3-Mini-128k-Instruct"
mcts:
  simulations_per_step: 8
random_seed: 42
# Add other minimal necessary fields for AppConfig to load without error
datasets: {}
features: {}
macro_actions: {}
self_play: {}
training: {}
long_run: {}
kenlm: {}
openai: {}
log_level: "INFO"
"""
        with open(test_config_dir / "default.yaml", "w") as f:
            f.write(dummy_yaml_content)
            print(f"Created dummy {test_config_dir / 'default.yaml'} for standalone test run.")

    unittest.main() 