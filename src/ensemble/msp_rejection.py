import numpy as np


def apply_msp_rejection(probabilities, threshold=0.5, unknown_label=5):
    max_probs = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    
    unknown_mask = max_probs < threshold
    predictions[unknown_mask] = unknown_label
    
    return predictions, max_probs


def calculate_rejection_metrics(y_true, predictions, unknown_label=5):
    known_mask = y_true != unknown_label
    unknown_mask = y_true == unknown_label
    
    n_known = known_mask.sum()
    n_unknown = unknown_mask.sum()
    
    known_correct = ((predictions[known_mask] == y_true[known_mask]) & 
                     (predictions[known_mask] != unknown_label)).sum()
    
    unknown_correct = (predictions[unknown_mask] == unknown_label).sum()
    
    known_acc = known_correct / n_known if n_known > 0 else 0
    unknown_recall = unknown_correct / n_unknown if n_unknown > 0 else 0
    
    total_correct = known_correct + unknown_correct
    total_acc = total_correct / len(y_true)
    
    metrics = {
        'known_accuracy': known_acc,
        'unknown_recall': unknown_recall,
        'total_accuracy': total_acc,
        'n_known': n_known,
        'n_unknown': n_unknown,
        'known_correct': known_correct,
        'unknown_correct': unknown_correct
    }
    
    return metrics


def find_best_threshold_f1(probabilities, y_true, unknown_label=5, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.9, 30)
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = None
    
    for threshold in thresholds:
        predictions, _ = apply_msp_rejection(probabilities, threshold, unknown_label)
        
        known_mask = y_true != unknown_label
        unknown_mask = y_true == unknown_label
        
        tp = (predictions[unknown_mask] == unknown_label).sum()
        fp = (predictions[known_mask] == unknown_label).sum()
        fn = (predictions[unknown_mask] != unknown_label).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    return best_threshold, best_metrics
