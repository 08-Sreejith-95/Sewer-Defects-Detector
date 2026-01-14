from omegaconf import OmegaConf

def load_config(path:str):
    """Load configuration from a YAML file."""
    cfg = OmegaConf.load(path)
    return cfg