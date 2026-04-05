import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from wide_resnet.config.config import get_config
from wide_resnet.models.wide_resnet_model import create_model
from utils.dataset import TestDataset
from utils.augmentation import get_val_transforms


def load_model(checkpoint_path, config=None):
    if config is None:
        config = get_config()
    
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    class_to_idx = checkpoint.get('class_to_idx', None)
    
    return model, class_to_idx


def extract_features(model, dataloader, device):
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            features = model.extract_features(images)
            
            all_features.append(features.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.numpy())
            else:
                all_labels.extend(labels)
    
    all_features = np.vstack(all_features)
    
    return all_features, all_labels


def predict(model, dataloader, device):
    all_predictions = []
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.numpy())
            else:
                all_labels.extend(labels)
    
    all_predictions = np.concatenate(all_predictions)
    all_probs = np.vstack(all_probs)
    
    return all_predictions, all_probs, all_labels


if __name__ == '__main__':
    config = get_config()
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print("Loading model...")
    model, class_to_idx = load_model(checkpoint_path, config)
    
    print("Loading test data...")
    test_dataset = TestDataset(
        data_dir='data/test',
        transform=get_val_transforms(config.image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print("Running inference...")
    predictions, probs, image_names = predict(model, test_loader, config.device)
    
    print(f"\nPredicted {len(predictions)} samples")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probs.shape}")
