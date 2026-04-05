import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import pickle
import xgboost as xgb
import pandas as pd
from torch.utils.data import DataLoader

from ensemble.config.config import get_config
from ensemble.feature_extractor import FeatureExtractor
from ensemble.msp_rejection import apply_msp_rejection
from utils.dataset import TestDataset
from utils.augmentation import get_val_transforms


def load_msp_threshold(config):
    threshold_path = Path(config.model_save_path).parent / 'msp_threshold.txt'
    if threshold_path.exists():
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Loaded MSP threshold: {threshold:.4f}")
        return threshold
    else:
        print(f"Threshold file not found, using default: {config.msp_threshold}")
        return config.msp_threshold


def predict_test_set(config=None):
    if config is None:
        config = get_config()
    
    print("Loading XGBoost model...")
    with open(config.model_save_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Loading feature extractor...")
    extractor = FeatureExtractor(config)
    
    idx_to_class = {idx: cls for cls, idx in extractor.class_to_idx.items()}
    idx_to_class[5] = config.unknown_class_label
    
    print("Loading test dataset...")
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
    
    print("Extracting features from test set...")
    features, image_names = extractor.extract_features(test_loader)
    
    print("\nPredicting with XGBoost...")
    dmatrix = xgb.DMatrix(features)
    probabilities = model.predict(dmatrix)
    
    threshold = load_msp_threshold(config)
    
    print(f"Applying MSP rejection with threshold {threshold:.4f}...")
    predictions, max_probs = apply_msp_rejection(probabilities, threshold, unknown_label=5)
    
    predicted_labels = [idx_to_class[pred] for pred in predictions]
    
    results_df = pd.DataFrame({
        'id': image_names,
        'label': predicted_labels,
        'max_prob': max_probs
    })
    
    print(f"\nPredicted {len(results_df)} samples")
    print(f"\nClass distribution:")
    print(results_df['label'].value_counts())
    
    return results_df, probabilities


def generate_submission(output_path='outputs/submissions/submission.csv'):
    config = get_config()
    
    results_df, _ = predict_test_set(config)
    
    submission_df = results_df[['id', 'label']].copy()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission_df.shape}")
    print(f"\nFirst few rows:")
    print(submission_df.head())
    
    return submission_df


if __name__ == '__main__':
    generate_submission()
