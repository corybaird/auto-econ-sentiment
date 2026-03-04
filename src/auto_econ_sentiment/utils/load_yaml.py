import yaml
import logging
import traceback
from pathlib import Path

def load_yaml_config(config_path="params.yaml"):
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logging.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}")
            return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        traceback.print_exc()
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading config: {e}")
        traceback.print_exc()
        raise