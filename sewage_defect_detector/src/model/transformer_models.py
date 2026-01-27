import timm
import torch.nn as nn

def build_vit_model(cfg):
    model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        num_classes=cfg.num_classes
    )
    
    in_features = model.head.in_features
    model.head = nn.Sequential(
                               nn.LayerNorm(in_features), 
                               nn.Linear(in_features, cfg.hidden_dim),
                               nn.GELU(),
                               nn.Dropout(cfg.drop_out),
                               nn.Linear(cfg.hidden_dim, cfg.num_classes)
                               )
    return model