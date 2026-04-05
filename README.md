# Face Spoofing Detection - Ensemble Model

A face spoofing detection system using an ensemble of transformer and CNN models with open-set rejection.

## Overview

This project implements a 6-class face spoofing detection system that trains on 5 known classes and uses Maximum Softmax Probability (MSP) thresholding to detect unknown attack types.

### Classes
- `realperson`: Genuine face images
- `fake_printed`: Print attack (photo paper)
- `fake_screen`: Screen attack (digital display)
- `fake_mask`: 3D mask or silicone mask attack
- `fake_mannequin`: Mannequin replica attack
- `fake_unknown`: Unknown spoofing attacks (detected via MSP threshold)

### Approach
1. Train 4 feature extractors (DINO, ViT, ResNet, Wide ResNet) on 5 known classes
2. Extract and concatenate features from all models
3. Train XGBoost meta-learner on concatenated features
4. Use MSP threshold at inference to reject unknown samples as `fake_unknown`

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- UV package manager

## Installation

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Or run commands with uv
uv run python <script.py>
```

## Project Structure

```
findit-data/
├── data/
│   ├── train/              # Original training data (6 class folders)
│   ├── test/               # Original test images
│   ├── Train/              # Preprocessed training split
│   ├── Validation/         # Preprocessed validation split
│   └── Test/               # Preprocessed test data
├── src/
│   ├── dino/               # DINO model module
│   ├── vit/                # ViT model module
│   ├── resnet/             # ResNet model module
│   ├── wide_resnet/        # Wide ResNet model module
│   ├── ensemble/           # XGBoost ensemble module
│   └── utils/              # Shared utilities
├── scripts/                # Training and preprocessing scripts
├── model/                  # Saved model weights
├── outputs/                # Checkpoints, logs, features, submissions
└── docs/                   # Documentation
```

## Usage

### 1. Prepare Data
```bash
uv run python scripts/prepare_data.py
```

### 2. Train Models
```bash
uv run python scripts/run_training.py
```

### 3. Generate Predictions
```bash
uv run python scripts/inference.py
```

## Model Architecture

- **Feature Extractors**: DINO, ViT, ResNet50, Wide ResNet (hybrid fine-tuning)
- **Meta-learner**: XGBoost (5-class classification)
- **Open-set Rejection**: MSP threshold for unknown detection

## License

MIT
