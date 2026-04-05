import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dino.config.config import get_config as get_dino_config
from dino.models.dino_model import create_model as create_dino_model
from vit.config.config import get_config as get_vit_config
from vit.models.vit_model import create_model as create_vit_model
from resnet.config.config import get_config as get_resnet_config
from resnet.models.resnet_model import create_model as create_resnet_model
from wide_resnet.config.config import get_config as get_wresnet_config
from wide_resnet.models.wide_resnet_model import create_model as create_wresnet_model

from utils.dataset import SpoofingDataset
from utils.augmentation import get_val_transforms


class FeatureExtractor:
    def __init__(self, ensemble_config):
        self.config = ensemble_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.models = {}
        self.class_to_idx = None
        
        self._load_models()
    
    def _load_models(self):
        print("Loading DINO model...")
        dino_config = get_dino_config()
        dino_model = create_dino_model(dino_config)
        checkpoint = torch.load(self.config.dino_checkpoint, map_location=self.device)
        dino_model.load_state_dict(checkpoint['model_state_dict'])
        dino_model = dino_model.to(self.device)
        dino_model.eval()
        self.models['dino'] = dino_model
        self.class_to_idx = checkpoint.get('class_to_idx', None)
        
        print("Loading ViT model...")
        vit_config = get_vit_config()
        vit_model = create_vit_model(vit_config)
        checkpoint = torch.load(self.config.vit_checkpoint, map_location=self.device)
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        vit_model = vit_model.to(self.device)
        vit_model.eval()
        self.models['vit'] = vit_model
        
        print("Loading ResNet model...")
        resnet_config = get_resnet_config()
        resnet_model = create_resnet_model(resnet_config)
        checkpoint = torch.load(self.config.resnet_checkpoint, map_location=self.device)
        resnet_model.load_state_dict(checkpoint['model_state_dict'])
        resnet_model = resnet_model.to(self.device)
        resnet_model.eval()
        self.models['resnet'] = resnet_model
        
        print("Loading Wide ResNet model...")
        wresnet_config = get_wresnet_config()
        wresnet_model = create_wresnet_model(wresnet_config)
        checkpoint = torch.load(self.config.wresnet_checkpoint, map_location=self.device)
        wresnet_model.load_state_dict(checkpoint['model_state_dict'])
        wresnet_model = wresnet_model.to(self.device)
        wresnet_model.eval()
        self.models['wide_resnet'] = wresnet_model
        
        print("All models loaded successfully!")
    
    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Extracting features'):
                images = images.to(self.device)
                
                dino_feats = self.models['dino'].extract_features(images).cpu().numpy()
                vit_feats = self.models['vit'].extract_features(images).cpu().numpy()
                resnet_feats = self.models['resnet'].extract_features(images).cpu().numpy()
                wresnet_feats = self.models['wide_resnet'].extract_features(images).cpu().numpy()
                
                combined_feats = np.concatenate([
                    dino_feats, vit_feats, resnet_feats, wresnet_feats
                ], axis=1)
                
                all_features.append(combined_feats)
                
                if isinstance(labels, torch.Tensor):
                    all_labels.append(labels.numpy())
                else:
                    all_labels.extend(labels)
        
        all_features = np.vstack(all_features)
        
        if len(all_labels) > 0 and isinstance(all_labels[0], np.ndarray):
            all_labels = np.concatenate(all_labels)
        
        return all_features, all_labels


def extract_and_save_features(ensemble_config, split='train'):
    import os
    os.makedirs(ensemble_config.features_cache_dir, exist_ok=True)
    
    extractor = FeatureExtractor(ensemble_config)
    
    if split == 'train':
        data_dir = 'data/Train'
    elif split == 'val':
        data_dir = 'data/Validation'
    else:
        raise ValueError(f"Unknown split: {split}")
    
    transform = get_val_transforms(ensemble_config.image_size)
    dataset = SpoofingDataset(
        data_dir=data_dir,
        transform=transform,
        class_mapping=extractor.class_to_idx
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=ensemble_config.batch_size,
        shuffle=False,
        num_workers=ensemble_config.num_workers,
        pin_memory=ensemble_config.pin_memory
    )
    
    print(f"\nExtracting features for {split} set...")
    features, labels = extractor.extract_features(dataloader)
    
    save_path = Path(ensemble_config.features_cache_dir) / f'{split}_features.npz'
    np.savez(save_path, features=features, labels=labels)
    
    print(f"Features saved to {save_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape if isinstance(labels, np.ndarray) else len(labels)}")
    
    return features, labels


if __name__ == '__main__':
    from ensemble.config.config import get_config
    
    config = get_config()
    extract_and_save_features(config, split='train')
    extract_and_save_features(config, split='val')
