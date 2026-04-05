import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import wandb

from dino.config.config import get_config
from dino.data.dataset import get_train_dataset, get_val_dataset
from dino.models.dino_model import create_model
from utils.wandb_utils import get_wandb_config, init_wandb, log_metrics, finish_wandb


def calculate_class_weights(dataset):
    distribution = dataset.get_class_distribution()
    total = sum(distribution.values())
    weights = []
    for cls_name in sorted(distribution.keys()):
        weight = total / (len(distribution) * distribution[cls_name])
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(config=None, use_wandb=True):
    if config is None:
        config = get_config()
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    if use_wandb:
        wandb_config = get_wandb_config()
        wandb_run = init_wandb(wandb_config, run_name='dino-training', model_type='dino')
        if wandb_run:
            wandb.config.update({
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'freeze_ratio': config.freeze_ratio,
                'model_name': config.model_name,
                'image_size': config.image_size
            })
    
    print("Loading datasets...")
    train_dataset = get_train_dataset(config)
    val_dataset = get_val_dataset(config, class_mapping=train_dataset.class_to_idx)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.class_to_idx}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(config.device)
    
    if use_wandb and wandb.run:
        wandb.watch(model, log='all', log_freq=100)
    
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs - config.warmup_epochs
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nTraining on {config.device}...")
    print(f"Model: {config.model_name}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if use_wandb and wandb.run:
            log_metrics({
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'epoch': epoch
            })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_to_idx': train_dataset.class_to_idx
            }, checkpoint_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")
    
    if use_wandb and wandb.run:
        wandb.run.summary['best_val_accuracy'] = best_val_acc
        finish_wandb()
    
    return model


if __name__ == '__main__':
    train_model()
