#!/bin/bash

for model in vit resnet wide_resnet; do
    echo "Updating $model training script..."
    
    sed -i '/^from tqdm import tqdm$/a import wandb' "src/$model/training/train.py"
    
    sed -i "/^from ${model}.models/a from utils.wandb_utils import get_wandb_config, init_wandb, log_metrics, finish_wandb" "src/$model/training/train.py"
    
    sed -i 's/def train_model(config=None):/def train_model(config=None, use_wandb=True):/' "src/$model/training/train.py"
    
done

echo "Done!"
