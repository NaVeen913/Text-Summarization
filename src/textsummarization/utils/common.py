import os
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any
from ensure import ensure_annotations

from src.textsummarization.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox (dict-like object)
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns size of a file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    logger.info(f"File size: {path} → {size_in_kb} KB")
    return f"{size_in_kb} KB"
