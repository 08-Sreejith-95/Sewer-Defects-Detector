import os
import pandas as pd
import torch

#to check if the code is running in a Kaggle environment
def is_kaggle():
    return os.path.exists('/kaggle/input')

#to compute class weights for imbalanced datasets only for training data
def compute_class_weights(cfg):
    from src.path import get_csv_path
    label_csv_path_train = get_csv_path(cfg, "train")
    data = pd.read_csv(label_csv_path_train)
    label_columns = data.columns[1:]  # Assuming first column is image names
    positive_counts = data[label_columns].sum()
    negative_counts = len(data) - positive_counts
    pos_weights = negative_counts / (positive_counts + 1e-6)  # Adding a small value to avoid division by zero
    pos_weights = pos_weights.clip(upper=20) 
    class_weights = torch.tensor(
        pos_weights.values,
        dtype = torch.float32)
    
    return class_weights


#To override config parameters with command-line arguments
def override_cfg(cfg, args):
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = args.weight_decay
    return cfg

