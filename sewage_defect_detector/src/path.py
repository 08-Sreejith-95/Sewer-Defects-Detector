import os
from src.utils import is_kaggle


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
    images_dir = (
        cfg["dataset"]["train_imgs"]
        if split == "train"
        else cfg["dataset"]["test_imgs"]
    )
    return os.path.join(
        get_dataset_root(cfg),
        images_dir
    )

def get_csv_path(cfg, split="train"):
    csv_name = (
        cfg["dataset"]["train_csv"]
        if split == "train"
        else cfg["dataset"]["test_csv"]
    )
    return os.path.join(get_dataset_root(cfg), csv_name)
