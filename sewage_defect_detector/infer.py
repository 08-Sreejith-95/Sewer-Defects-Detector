from src.utils import parse_args
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.config.config import load_config
from src.datasets.sewer_ml_dataset import SewerMLDataset
from src.model.transformer_models import build_vit_model
import sys
from src.utils.arg_parser import parse_args

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def infer():
    args = parse_args()
    
    # Load OmegaConf config
    cfg = load_config(args.config)
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Create Dataset & DataLoader for test set
    test_ds = SewerMLDataset(cfg=cfg, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )
    
    # Load model
    model = build_vit_model(cfg.dataset.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    all_probs = []
    all_names = []
    
    # Inference loop
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)  # multi-label probabilities
            all_probs.append(probs.cpu())
            all_names.extend(img_names)
    
    # Combine predictions
    probs = torch.cat(all_probs).numpy()
    class_names = cfg.dataset.class_names
    
    # Create submission DataFrame
    submission = pd.DataFrame(probs, columns=class_names)
    submission.insert(0, "image_name", all_names)
    
    # Save CSV
    submission.to_csv(args.submission_name, index=False)
    print(f"Submission saved as {args.submission_name}")

if __name__ == "__main__":
    infer()
