"""
Loads and provides access to the project configuration defined in configs/default.yaml.

This module reads the YAML configuration file once upon import and makes it
available as a module-level dictionary `cfg`.

Usage:
    from src.config import cfg
    learning_rate = cfg['training']['adamw']['lr']
"""

import yaml
from pathlib import Path

# Determine the path to the default config file relative to this script's location.
# Assumes config.py is in src/ and default.yaml is in configs/
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loads the YAML configuration file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # Handle empty file case
             return {}
        return config_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        raise

# Load the configuration once when the module is imported.
cfg = load_config()

# Example of accessing a value (optional, for demonstration)
if __name__ == '__main__':
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(f"SLM Model Name: {cfg.get('slm', {}).get('model_name', 'Not Found')}")
    print(f"AdamW Learning Rate: {cfg.get('training', {}).get('adamw', {}).get('lr', 'Not Found')}") 