import torch
import torch.nn as nn
import timm


class ResNetModel(nn.Module):
    def __init__(self, config):
        super(ResNetModel, self).__init__()
        self.config = config
        
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=True,
            num_classes=0
        )
        
        self._freeze_layers()
        
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
    
    def _freeze_layers(self):
        total_params = len(list(self.backbone.parameters()))
        freeze_count = int(total_params * self.config.freeze_ratio)
        
        for idx, param in enumerate(self.backbone.parameters()):
            if idx < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return features


def create_model(config):
    model = ResNetModel(config)
    return model
