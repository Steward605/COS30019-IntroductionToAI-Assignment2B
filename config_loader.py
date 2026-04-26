from pathlib import Path
import json

CONFIG_FILE = Path("config.json")

def load_config(config_file=CONFIG_FILE):
    if not config_file.exists():
        raise FileNotFoundError("config.json not found. Please create config.json in the project root folder.")
    with open(config_file, "r", encoding="utf-8") as file:
        return json.load(file)

def get_config_value(config, key_path, default=None):
    current_value = config
    for key in key_path:
        if not isinstance(current_value, dict) or key not in current_value:
            return default
        current_value = current_value[key]
    return current_value