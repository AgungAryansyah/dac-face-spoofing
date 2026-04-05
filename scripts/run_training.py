import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse


def train_all_models():
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    print("\n[1/4] Training DINO model...")
    from dino.training.train import train_model as train_dino
    train_dino()
    
    print("\n[2/4] Training ViT model...")
    from vit.training.train import train_model as train_vit
    train_vit()
    
    print("\n[3/4] Training ResNet model...")
    from resnet.training.train import train_model as train_resnet
    train_resnet()
    
    print("\n[4/4] Training Wide ResNet model...")
    from wide_resnet.training.train import train_model as train_wresnet
    train_wresnet()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - ALL 4 MODELS")
    print("=" * 60)


def train_ensemble():
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE")
    print("=" * 60)
    
    print("\n[1/3] Extracting features...")
    from ensemble.feature_extractor import extract_and_save_features
    from ensemble.config.config import get_config
    
    config = get_config()
    extract_and_save_features(config, split='train')
    extract_and_save_features(config, split='val')
    
    print("\n[2/3] Training XGBoost...")
    from ensemble.training.train_xgboost import train_xgboost
    train_xgboost(config)
    
    print("\n[3/3] Tuning MSP threshold...")
    from ensemble.tune_threshold import tune_threshold_with_unknown_samples
    tune_threshold_with_unknown_samples()
    
    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train face spoofing detection models')
    parser.add_argument(
        '--stage',
        choices=['models', 'ensemble', 'all'],
        default='all',
        help='Training stage: models (4 feature extractors), ensemble (XGBoost), or all'
    )
    
    args = parser.parse_args()
    
    if args.stage in ['models', 'all']:
        train_all_models()
    
    if args.stage in ['ensemble', 'all']:
        train_ensemble()
    
    print("\n✓ Training pipeline complete!")


if __name__ == '__main__':
    main()
