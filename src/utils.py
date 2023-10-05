import os
import json
from typing import Dict, Any

def import_config_dict(config_path: str='config/config.json') -> Dict[str, Any]:
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def import_params(model_name: str, config_path: str='config/model') -> Dict[str, Any]:
    with open(os.path.join(config_path, model_name + '.json')) as config_file:
        config = json.load(config_file)
    return config
