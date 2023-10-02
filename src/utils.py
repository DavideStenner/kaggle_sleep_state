import json
from typing import Dict, Any

def import_config_dict(config_path: str='config/config.json') -> Dict[str, Any]:
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

