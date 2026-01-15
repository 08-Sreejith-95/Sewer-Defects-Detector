import os
from src.utils.utils import is_kaggle


def get_data_root(cfg):
    if is_kaggle():
        return cfg["env"]["kaggle_data_root"]
    return cfg["env"]["local_data_root"]

def get_dataset_root(cfg):
    return os.path.join(
        get_data_root(cfg),
        cfg["dataset"]["name"]
    )

def get_image_dir(cfg, split="train"):
    if split in ("train", "val", "valid", "validation"):
        images_dir = cfg["dataset"]["train_imgs"]
    elif split == "test":
        images_dir = cfg["dataset"]["test_imgs"]
    else:
        raise ValueError(f"Unknown split: {split}")

    return os.path.join(get_dataset_root(cfg), images_dir)

def get_csv_path(cfg, split="train"):
    csv_name = (
        cfg["dataset"]["train_csv"]
        if split == "train"
        else cfg["dataset"]["test_csv"]
    )
    return os.path.join(get_dataset_root(cfg), csv_name)
