import timm
import torch.nn as nn


#to do:- change head to default for checking the onnx runtime output(), because the checkpoint model is older version default. with default convnxttiny settings

def build_vit_model(cfg):
    #print(f"loaded config: {model_cfg}")
    if cfg.modified_head:
        print("Using modified head with hidden_dim =", cfg.model.hidden_dim)
        model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        num_classes= 0) # remove default head
        in_features = model.num_features
        model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(1),
                               nn.LayerNorm(in_features), 
                               nn.Linear(in_features, cfg.model.hidden_dim),
                               nn.GELU(),
                               nn.Dropout(cfg.model.drop_out),
                               nn.Linear(cfg.model.hidden_dim, cfg.dataset.num_classes)
                               )
    else:
        print("Using default head with num_classes =", cfg.dataset.num_classes)
        model = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=cfg.dataset.num_classes
        )
    return model