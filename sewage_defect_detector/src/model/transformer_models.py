import timm
import torch.nn as nn

def build_vit_model(cfg):
    #print(f"loaded config: {model_cfg}")
    model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        num_classes= 0, # we'll replace the head,
    )
    
    in_features = model.num_features
    model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(1),
                               nn.LayerNorm(in_features), 
                               nn.Linear(in_features, cfg.model.hidden_dim),
                               nn.GELU(),
                               nn.Dropout(cfg.model.drop_out),
                               nn.Linear(cfg.model.hidden_dim, cfg.dataset.num_classes)
                               )
    return model