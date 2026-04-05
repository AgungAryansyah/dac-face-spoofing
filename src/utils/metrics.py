import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, class_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class
    }
    
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names)
        metrics['classification_report'] = report
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_metrics(metrics, class_names=None):
    print("=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print()
    
    if class_names and 'f1_per_class' in metrics:
        print("Per-Class Metrics:")
        print("-" * 50)
        for i, name in enumerate(class_names):
            print(f"{name:20s} - F1: {metrics['f1_per_class'][i]:.4f}, "
                  f"Precision: {metrics['precision_per_class'][i]:.4f}, "
                  f"Recall: {metrics['recall_per_class'][i]:.4f}, "
                  f"Support: {metrics['support_per_class'][i]}")
    
    if 'classification_report' in metrics:
        print()
        print("Classification Report:")
        print("-" * 50)
        print(metrics['classification_report'])
