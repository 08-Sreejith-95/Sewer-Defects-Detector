
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.transformer_models import build_vit_model
from src.config.config import load_config
from src.utils.arg_parser import parse_args
from src.path import get_image_dir, get_csv_path

args = parse_args()
cfg = load_config(args.config)


#Function to visualize heatmaps using GradCAM for a given image and model for evaluation
def gradcam_visualize(image_path, model):
    
    target_layers = [model.stages[-1].blocks[-1].conv_dw]

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.523, 0.453, 0.345],
                    std=[0.210, 0.199, 0.154])
    ])

    image = Image.open(image_path).convert("RGB")
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)  # multi-label so sigmoid not softmax
        predicted_classes = (probs > 0.4).nonzero(as_tuple=True)[1]
        print("Predicted classes:", [cfg.dataset.class_names[i.item()] for i in predicted_classes])
    target_class_idx = probs.squeeze().argmax().item()
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    visualization = show_cam_on_image(
        rgb_img.astype(np.float32), grayscale_cam, use_rgb=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(visualization)
    axes[1].set_title(f"GradCAM — Class: {cfg.dataset.class_names[target_class_idx]}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"gradcam_{cfg.dataset.class_names[target_class_idx]}.png", dpi=150)
    plt.show()
    
if __name__ == "__main__":
    model = build_vit_model(cfg)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

# Then use the top predicted class for GradCAM
    model.eval()
    sample_image = "/kaggle/working/Sewer-Defects-Detector/sewage_defect_detector/visualization/PB_RB_OB_FS.png"  # To do:add configuration option for this path
    gradcam_visualize(sample_image, model)
