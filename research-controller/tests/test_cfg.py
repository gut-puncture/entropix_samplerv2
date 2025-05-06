"""
Tests for the configuration loading in src/config.py.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Attempt to import the config after modifying path
try:
    from src.config import cfg, load_config, CONFIG_PATH
except ImportError as e:
    pytest.fail(f"Failed to import config module: {e}\nPython path: {sys.path}")

def test_config_exists_and_is_dict():
    """Test that the config file loads and is a dictionary."""
    assert cfg is not None, "Config object 'cfg' should be loaded."
    assert isinstance(cfg, dict), "Config object 'cfg' should be a dictionary."
    assert len(cfg) > 0, "Config dictionary should not be empty."

def test_required_top_level_keys():
    """Test that essential top-level keys are present."""
    required_keys = ['environment', 'slm', 'mcts', 'training', 'reward', 'network']
    for key in required_keys:
        assert key in cfg, f"Missing required top-level key: '{key}'"

def test_required_nested_keys():
    """Test that essential nested keys are present."""
    assert 'model_name' in cfg.get('slm', {}), "Missing slm.model_name"
    assert 'load_bits' in cfg.get('slm', {}), "Missing slm.load_bits"
    assert 'c_puct' in cfg.get('mcts', {}), "Missing mcts.c_puct"
    assert 'adamw' in cfg.get('training', {}), "Missing training.adamw"
    assert 'lr' in cfg.get('training', {}).get('adamw', {}), "Missing training.adamw.lr"
    assert 'weights' in cfg.get('reward', {}), "Missing reward.weights"
    assert 'g_eval_coherence' in cfg.get('reward', {}).get('weights', {}), "Missing reward.weights.g_eval_coherence"

def test_load_config_raises_filenotfound():
    """Test that load_config raises FileNotFoundError for a non-existent file."""
    non_existent_path = Path("non_existent_config.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(non_existent_path)

# You can add more tests here to check specific values or types if needed. 