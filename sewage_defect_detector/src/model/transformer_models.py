import timm
import torch.nn as nn

def build_vit_model(model_cfg):
    #print(f"loaded config: {model_cfg}")
    model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        num_classes= model_cfg["num_classes"],
    )
    
    in_features = model.head.in_features
    model.head = nn.Sequential(
                               nn.LayerNorm(in_features), 
                               nn.Linear(in_features, model_cfg["hidden_dim"]),
                               nn.GELU(),
                               nn.Dropout(model_cfg["drop_out"]),
                               nn.Linear(model_cfg["hidden_dim"], model_cfg["num_classes"])
                               )
    return model