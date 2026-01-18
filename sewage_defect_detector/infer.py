
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from src.path import get_image_dir, get_csv_path

from src.config.config import load_config
from src.datasets.sewer_ml_dataset import SewerMLDataset
from src.model.transformer_models import build_vit_model
import sys
from src.utils.arg_parser import parse_args



# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def infer():
    args = parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)

    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- Transforms (MUST match training) ----
    test_transforms = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.std
        )
    ])

    # ---- Load test CSV ----
    test_csv = get_csv_path(cfg, "test")
    test_df = pd.read_csv(test_csv)

    # ---- Dataset & Loader ----
    test_ds = SewerMLDataset(
        cfg=cfg,
        split="test",
        df=test_df,
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,          # IMPORTANT for Colab
        pin_memory=True
    )

    # ---- Load model ----
    model = build_vit_model(cfg.dataset.num_classes)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    model.to(device)
    model.eval()

    all_probs = []
    all_names = []

    # ---- Inference ----
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            thresholds = np.array(cfg.dataset.CIW)
            preds = (probs >= thresholds).astype(int)

            all_probs.append(preds)
            all_names.extend(img_names)

    # ---- Post-processing ----
    probs = torch.cat(all_probs).numpy()
    class_names = cfg.dataset.class_names

    submission = pd.DataFrame(probs, columns=class_names)
    submission.insert(0, "Filename", all_names)

    submission.to_csv(f"/kaggle/working/{args.submission_name}", index=False)
    print(f" Submission saved at: /kaggle/working/{args.submission_name}")


if __name__ == "__main__":
    infer()
