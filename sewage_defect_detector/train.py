import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

#os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_MODE"] = "disabled"


from src.config.config import load_config
from src.datasets.sewer_ml_dataset import SewerMLDataset
from src.model.transformer_models import build_vit_model
from src.utils.utils import compute_class_weights, override_cfg
from src.utils.arg_parser import parse_args# your existing argparse
from src.path import get_image_dir, get_csv_path
import sys
import wandb

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#wandb.login()

def train():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_cfg(cfg, args)
    
    wandb.init(
    project="sewage-ml",
    name=args.run_name,
    config={
        "model": "convnext_tiny",
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "epochs": cfg.training.epochs,
        "img_size": cfg.dataset.img_size,
        "num_classes": cfg.dataset.num_classes,
        "loss": "BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "amp": True
    }
)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # --- Load full train CSV ---
    data_path = get_csv_path(cfg, "train")
    df = pd.read_csv(data_path)
    
    # --- Train/val split ---
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True# multi-label stratify
    )
    
    train_transforms = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    ])
    # --- Datasets ---
    train_ds = SewerMLDataset(cfg=cfg, split="train", df=train_df, transform=train_transforms)
    val_ds   = SewerMLDataset(cfg=cfg, split="val", df=val_df, transform=val_transforms)
    
    #debugging
    #print(train_ds.data[train_ds.label_cols].dtypes)
    #print(train_ds.data[train_ds.label_cols].head())
    
    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )
    
    # --- Model ---
    model = build_vit_model(cfg.dataset.num_classes).to(device)
    
    # Resume checkpoint if provided
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed model from {args.resume}")
    
    # --- Loss, optimizer ---
    class_weights = compute_class_weights(cfg).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scaler = GradScaler()
    
    # --- Output folder ---
    os.makedirs(f"outputs/{args.run_name}", exist_ok=True)

    best_f1_micro = float("-inf")
    best_model_path = f"outputs/{args.run_name}/model_best.pt"
    
    # --- Training loop ---
    for epoch in range(cfg.training.epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ---- Validate ----
        model.eval()
        #total_val_loss = 0.0
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                #total_val_loss += loss.item()
        #avg_val_loss = total_val_loss / len(val_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        avg_val_loss = np.mean(val_losses)
        val_f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        print(f"Epoch [{epoch+1}/{cfg.training.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f}"
              f" Val F1 Micro: {val_f1_micro:.4f} "
              f"Val F1 Macro: {val_f1_macro:.4f} "
              f"Val F1 Samples: {val_f1_samples:.4f} "
              f"Val Precision: {val_precision:.4f} "
              f"Val Recall: {val_recall:.4f}")
        
        # ---- Save best checkpoint only ----
        if val_f1_micro > best_f1_micro:
            best_f1_micro = val_f1_micro
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")
        
        
        wandb.log({
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/f1_micro": val_f1_micro,
                "val/f1_macro": val_f1_macro,
                "val/f1_samples": val_f1_samples,
                "val/precision": val_precision,
                "val/recall": val_recall
                })

    
    print(f"Training finished. Best model saved at: {best_model_path}")
    wandb.finish()

if __name__ == "__main__":
    train()
