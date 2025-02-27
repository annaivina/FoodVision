import yaml
import os

def load_config(config_path=""):

    if not config_path:
        raise ValueError("Config path is not provided")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")
    

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

