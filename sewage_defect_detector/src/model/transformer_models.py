import timm
import torch.nn as nn

def build_vit_model(num_classes):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model