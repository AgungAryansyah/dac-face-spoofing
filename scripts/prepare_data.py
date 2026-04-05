import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random


def prepare_data(
    source_dir='data/train',
    train_dir='data/Train',
    val_dir='data/Validation',
    val_split=0.2,
    exclude_classes=['fake_unknown'],
    random_seed=42
):
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    class_files = defaultdict(list)
    
    for class_name in os.listdir(source_path):
        class_path = source_path / class_name
        if not class_path.is_dir():
            continue
        
        if class_name in exclude_classes:
            print(f"Excluding class: {class_name} (open-set)")
            continue
        
        files = list(class_path.glob('*'))
        files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        class_files[class_name] = files
        
        print(f"Class {class_name}: {len(files)} images")
    
    print(f"\nSplitting data with validation ratio: {val_split}")
    
    for class_name, files in class_files.items():
        train_class_path = train_path / class_name
        val_class_path = val_path / class_name
        
        train_class_path.mkdir(parents=True, exist_ok=True)
        val_class_path.mkdir(parents=True, exist_ok=True)
        
        train_files, val_files = train_test_split(
            files,
            test_size=val_split,
            random_state=random_seed,
            shuffle=True
        )
        
        for file_path in train_files:
            dest = train_class_path / file_path.name
            shutil.copy2(file_path, dest)
        
        for file_path in val_files:
            dest = val_class_path / file_path.name
            shutil.copy2(file_path, dest)
        
        print(f"  {class_name}: {len(train_files)} train, {len(val_files)} val")
    
    print("\nData preparation complete!")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")


if __name__ == '__main__':
    prepare_data()
