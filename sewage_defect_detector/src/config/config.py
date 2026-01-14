from omegaconf import OmegaConf

def load_config(path = "configs/configs.yaml"):
    """Load configuration from a YAML file."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)