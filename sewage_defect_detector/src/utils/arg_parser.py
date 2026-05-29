import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt on Sewer-ML")
    
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    parser.add_argument("--checkpoint", type=str, default = None, help="Path to trained model")#use this checkpoint for onnx export args
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--submission-name", type=str, default="submission.csv")
    parser.add_argument("--run-name", type=str, default="run1", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Whether to log to Weights & Biases")
    ##onnx export args
    
    parser.add_argument("--onnx-output", default="sewage_defect_detector/onnx_models/sewer_model.onnx", help="Path to save ONNX model")
    
    #onnx inference args
    parser.add_argument("--model", help="Path to .onnx model for inference")
    parser.add_argument("--test-image-dir", help="Directory of test images for ONNX inference")
    parser.add_argument("--threshold", type=float, default=0.4, help="Probability threshold for classifying defects in ONNX inference")
    
    return parser.parse_args()
