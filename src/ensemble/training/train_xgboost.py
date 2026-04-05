import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, classification_report

from ensemble.config.config import get_config
from ensemble.feature_extractor import extract_and_save_features


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


def train_xgboost(config=None):
    if config is None:
        config = get_config()
    
    os.makedirs(Path(config.model_save_path).parent, exist_ok=True)
    
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
    
    print("\nSaving model...")
    with open(config.model_save_path, 'wb') as f:
        pickle.dump(bst, f)
    print(f"Model saved to {config.model_save_path}")
    
    return bst


if __name__ == '__main__':
    train_xgboost()
