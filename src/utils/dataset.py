import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SpoofingDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_mapping=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        if class_mapping is None:
            classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_mapping
        
        for class_name in self.class_to_idx.keys():
            class_path = self.data_dir / class_name
            if not class_path.exists():
                continue
            
            for img_path in class_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label
    
    def get_class_distribution(self):
        distribution = {cls_name: 0 for cls_name in self.class_to_idx.keys()}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.samples = []
        for img_path in self.data_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.samples.append(str(img_path))
        
        self.samples = sorted(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        img_name = Path(img_path).stem
        
        return image, img_name
