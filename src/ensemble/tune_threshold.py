import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import pickle
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
import wandb

from ensemble.config.config import get_config
from ensemble.feature_extractor import FeatureExtractor
from ensemble.msp_rejection import find_best_threshold_f1, calculate_rejection_metrics, apply_msp_rejection
from utils.dataset import SpoofingDataset
from utils.augmentation import get_val_transforms
from utils.wandb_utils import get_wandb_config, init_wandb, log_metrics, finish_wandb


def tune_threshold_with_unknown_samples(use_wandb=True):
    config = get_config()
    
    if use_wandb:
        wandb_config = get_wandb_config()
        wandb_run = init_wandb(wandb_config, run_name='msp-threshold-tuning', model_type='threshold')
    
    print("Loading XGBoost model...")
    with open(config.model_save_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Loading feature extractor...")
    extractor = FeatureExtractor(config)
    
    known_classes = extractor.class_to_idx.copy()
    unknown_label = len(known_classes)
    
    print("\nLoading validation set (known classes)...")
    val_dataset_known = SpoofingDataset(
        data_dir='data/Validation',
        transform=get_val_transforms(config.image_size),
        class_mapping=known_classes
    )
    
    val_loader_known = DataLoader(
        val_dataset_known,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print("Extracting features from known validation samples...")
    known_features, known_labels = extractor.extract_features(val_loader_known)
    
    print("\nLoading unknown class samples (fake_unknown)...")
    unknown_dataset = SpoofingDataset(
        data_dir='data/train',
        transform=get_val_transforms(config.image_size),
        class_mapping={'fake_unknown': 5}
    )
    
    unknown_loader = DataLoader(
        unknown_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print("Extracting features from unknown samples...")
    unknown_features, _ = extractor.extract_features(unknown_loader)
    unknown_labels = np.full(len(unknown_features), unknown_label)
    
    print(f"\nKnown samples: {len(known_labels)}")
    print(f"Unknown samples: {len(unknown_labels)}")
    
    all_features = np.vstack([known_features, unknown_features])
    all_labels = np.concatenate([known_labels, unknown_labels])
    
    print("\nPredicting probabilities...")
    dmatrix = xgb.DMatrix(all_features)
    probabilities = model.predict(dmatrix)
    
    print("\nTuning MSP threshold...")
    best_threshold, metrics = find_best_threshold_f1(
        probabilities, 
        all_labels, 
        unknown_label=unknown_label
    )
    
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"F1-score for unknown class: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print("\nEvaluating with best threshold...")
    predictions, max_probs = apply_msp_rejection(probabilities, best_threshold, unknown_label)
    eval_metrics = calculate_rejection_metrics(all_labels, predictions, unknown_label)
    
    print(f"\nFinal metrics:")
    print(f"Known accuracy: {eval_metrics['known_accuracy']:.4f}")
    print(f"Unknown recall: {eval_metrics['unknown_recall']:.4f}")
    print(f"Total accuracy: {eval_metrics['total_accuracy']:.4f}")
    
    if use_wandb and wandb.run:
        log_metrics({
            'threshold/best_threshold': best_threshold,
            'threshold/f1_score': metrics['f1_score'],
            'threshold/precision': metrics['precision'],
            'threshold/recall': metrics['recall'],
            'threshold/known_accuracy': eval_metrics['known_accuracy'],
            'threshold/unknown_recall': eval_metrics['unknown_recall'],
            'threshold/total_accuracy': eval_metrics['total_accuracy']
        })
        finish_wandb()
    
    config_path = Path(config.model_save_path).parent / 'msp_threshold.txt'
    with open(config_path, 'w') as f:
        f.write(f"{best_threshold}\n")
    print(f"\nThreshold saved to {config_path}")
    
    return best_threshold, metrics


if __name__ == '__main__':
    tune_threshold_with_unknown_samples()
