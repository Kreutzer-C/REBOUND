import json
import logging
import os
from types import SimpleNamespace

def load_config_as_namespace(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    
    return dict_to_namespace(config_dict)


def convert_namespace_to_dict(ns):
    return {k: v for k, v in ns.__dict__.items() if not k.startswith('_')}


def get_logger(name, log_file=None, level=logging.INFO):
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger