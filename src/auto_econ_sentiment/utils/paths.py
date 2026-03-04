import os
from pathlib import Path

def get_data_path(filename):
    base_path = Path(__file__).parent.parent / "data"
    return str(base_path / filename)

def get_default_params_path():
    return str(Path(__file__).parent.parent.parent.parent / "params.yaml")
