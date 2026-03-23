import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on Sewer-ML")
    
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint", type=str, default = None, help="Path to trained model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--submission-name", type=str, default="submission.csv")
    parser.add_argument("--run-name", type=str, default="run1", help="Experiment name")
    
    return parser.parse_args()
