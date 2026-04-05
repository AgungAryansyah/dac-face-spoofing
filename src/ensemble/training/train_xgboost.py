import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, classification_report
import wandb

from ensemble.config.config import get_config
from ensemble.feature_extractor import extract_and_save_features
from utils.wandb_utils import get_wandb_config, init_wandb, log_metrics, finish_wandb


def load_features(config, split='train'):
    feature_path = Path(config.features_cache_dir) / f'{split}_features.npz'
    
    if not feature_path.exists():
        print(f"Features not found at {feature_path}")
        print(f"Extracting features for {split} set...")
        features, labels = extract_and_save_features(config, split=split)
    else:
        print(f"Loading features from {feature_path}")
        data = np.load(feature_path)
        features = data['features']
        labels = data['labels']
    
    return features, labels


def train_xgboost(config=None, use_wandb=True):
    if config is None:
        config = get_config()
    
    os.makedirs(Path(config.model_save_path).parent, exist_ok=True)
    
    if use_wandb:
        wandb_config = get_wandb_config()
        wandb_run = init_wandb(wandb_config, run_name='xgboost-ensemble', model_type='ensemble')
        if wandb_run:
            wandb.config.update(config.xgboost_params)
    
    print("Loading training features...")
    X_train, y_train = load_features(config, split='train')
    
    print("Loading validation features...")
    X_val, y_val = load_features(config, split='val')
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evallist = [(dtrain, 'train'), (dval, 'val')]
    
    print("\nTraining XGBoost...")
    print(f"Parameters: {config.xgboost_params}")
    
    bst = xgb.train(
        config.xgboost_params,
        dtrain,
        num_boost_round=config.xgboost_params['n_estimators'],
        evals=evallist,
        verbose_eval=10
    )
    
    print("\nEvaluating on validation set...")
    y_pred_proba = bst.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    if use_wandb and wandb.run:
        log_metrics({
            'ensemble/val_accuracy': accuracy,
            'ensemble/feature_dim': X_train.shape[1]
        })
    
    print("\nSaving model...")
    with open(config.model_save_path, 'wb') as f:
        pickle.dump(bst, f)
    print(f"Model saved to {config.model_save_path}")
    
    if use_wandb and wandb.run:
        wandb.run.summary['val_accuracy'] = accuracy
        finish_wandb()
    
    return bst


if __name__ == '__main__':
    train_xgboost()
