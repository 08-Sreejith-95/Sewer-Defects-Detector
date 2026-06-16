
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


#Model Interpretibility with GradCAM 
def gradcam_visualize(image_path, model, save_dir="/kaggle/working"):
    
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

    # --- get all predicted classes ---
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).squeeze()

    predicted_indices = (probs > 0.5).nonzero(as_tuple=False).squeeze(1)

    # fallback: if nothing passes threshold, take top-1
    if len(predicted_indices) == 0:
        predicted_indices = [probs.argmax().item()]
        print("No class above threshold.--------using top-1 prediction------")
    
    predicted_names  = [cfg.dataset.class_names[i.item()] for i in predicted_indices]
    predicted_scores = [probs[i].item() for i in predicted_indices]

    print(f"Predicted {len(predicted_indices)} defect(s):")
    for name, score in zip(predicted_names, predicted_scores):
        print(f"  {name}: {score:.3f}")

    # --- build grid layout ---
    # Row 1: original image + all GradCAM heatmaps for comparison
    # Row 2: score bar chart summary
    n_defects  = len(predicted_indices)
    n_cols     = n_defects + 1          # +1 for original image
    fig        = plt.figure(figsize=(4 * n_cols, 9))

    # -- top row: original image + GradCAMs --
    axes_imgs = [fig.add_subplot(2, n_cols, i + 1) for i in range(n_cols)]

    # original image
    axes_imgs[0].imshow(image)
    axes_imgs[0].set_title("Input Image", fontsize=11, fontweight="bold", pad=8)
    axes_imgs[0].axis("off")

    # one GradCAM per predicted class
    cam = GradCAM(model=model, target_layers=target_layers)

    for plot_idx, (class_idx, class_name, score) in enumerate(
        zip(predicted_indices, predicted_names, predicted_scores)
    ):
        targets       = [ClassifierOutputTarget(class_idx.item() 
                         if hasattr(class_idx, 'item') else class_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        visualization = show_cam_on_image(
            rgb_img.astype(np.float32), grayscale_cam, use_rgb=True
        )

        ax = axes_imgs[plot_idx + 1]
        ax.imshow(visualization)
        ax.set_title(
            f"{class_name}\nscore: {score:.3f}",
            fontsize=10,
            fontweight="bold",
            pad=6,
            color="darkred" if score > 0.7 else "darkorange" if score > 0.5 else "gray"
        )
        ax.axis("off")

    # -- bottom row: score bar chart spanning full width --
    ax_bar = fig.add_subplot(2, 1, 2)
    colors = [
        "#d32f2f" if s > 0.7 else "#f57c00" if s > 0.5 else "#388e3c"
        for s in predicted_scores
    ]
    bars = ax_bar.barh(predicted_names, predicted_scores, color=colors, height=0.5)

    # add score labels on bars
    for bar, score in zip(bars, predicted_scores):
        ax_bar.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center", ha="left", fontsize=10, fontweight="bold"
        )

    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Confidence Score (sigmoid)", fontsize=10)
    ax_bar.set_title("Predicted Defect Scores", fontsize=11, fontweight="bold")
    ax_bar.axvline(x=0.4, color="gray", linestyle="--", linewidth=1, label="threshold=0.4")
    ax_bar.legend(fontsize=9)
    ax_bar.invert_yaxis()  # highest score at top

    # color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="High (>0.7)"),
        Patch(facecolor="#f57c00", label="Medium (0.5–0.7)"),
        Patch(facecolor="#388e3c", label="Low (0.4–0.5)"),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.suptitle(
        f"Multi-Label GradCAM — {n_defects} Defect(s) Detected",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    # --- save ---
    os.makedirs(save_dir, exist_ok=True)
    save_name = "_".join(predicted_names)
    save_path = os.path.join(save_dir, f"gradcam_{save_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")
if __name__ == "__main__":
    model = build_vit_model(cfg)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

# Then use the top predicted class for GradCAM
    model.eval()
    sample_image = "/kaggle/working/Sewer-Defects-Detector/sewage_defect_detector/visualization/defect_class_multiple.jpg"  # To do:add configuration option for this path
    gradcam_visualize(sample_image, model)
