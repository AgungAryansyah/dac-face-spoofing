import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from utils.dataset import SpoofingDataset
from utils.augmentation import get_train_transforms, get_val_transforms


def get_train_dataset(config, class_mapping=None):
    transform = get_train_transforms(image_size=config.image_size)
    dataset = SpoofingDataset(
        data_dir='data/Train',
        transform=transform,
        class_mapping=class_mapping
    )
    return dataset


def get_val_dataset(config, class_mapping):
    transform = get_val_transforms(image_size=config.image_size)
    dataset = SpoofingDataset(
        data_dir='data/Validation',
        transform=transform,
        class_mapping=class_mapping
    )
    return dataset
