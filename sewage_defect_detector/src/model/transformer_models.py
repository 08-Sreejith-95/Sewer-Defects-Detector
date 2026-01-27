import timm
import torch.nn as nn

def build_vit_model(cfg):
    #print(f"loaded config: {model_cfg}")
    model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        num_classes= cfg.dataset.num_classes,
    )
    
    in_features = model.head.in_features
    model.head = nn.Sequential(
                               nn.LayerNorm(in_features), 
                               nn.Linear(in_features, cfg.model.hidden_dim),
                               nn.GELU(),
                               nn.Dropout(cfg.model.dropout),
                               nn.Linear(cfg.model.hidden_dim, cfg.dataset.num_classes)
                               )
    return model