import os


WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'face-spoofing-detection')
WANDB_ENTITY = os.getenv('WANDB_ENTITY', None)
WANDB_ENABLED = os.getenv('WANDB_ENABLED', 'true').lower() == 'true'


WANDB_TAGS = [
    'ensemble',
    'face-spoofing',
    'open-set',
    'dino',
    'vit',
    'resnet',
    'xgboost'
]


WANDB_NOTES = """
Face Spoofing Detection using Ensemble of Transformers and CNNs

Architecture:
- DINO (ViT-Base): 768-dim features
- ViT (ViT-Base): 768-dim features
- ResNet50: 2048-dim features
- Wide ResNet50-2: 2048-dim features

Ensemble:
- XGBoost meta-learner on 5,632-dim concatenated features
- MSP threshold for open-set rejection (fake_unknown detection)

Training Strategy:
- 5-class closed-set training
- Hybrid fine-tuning (75% frozen)
- Open-set detection via Maximum Softmax Probability
"""
