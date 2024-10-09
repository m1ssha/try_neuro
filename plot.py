import numpy as np


def precision_score(y_true, y_pred, average='weighted'):
    """Вычисление точности (precision) для многоклассовой классификации."""
    classes = np.unique(y_true)
    precision_per_class = []
    total_samples = len(y_true)
    
    for cls in classes:
        true_positives = np.sum((y_pred == cls) & (y_true == cls))
        predicted_positives = np.sum(y_pred == cls)
        
        if predicted_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / predicted_positives
        
        if average == 'weighted':
            weight = np.sum(y_true == cls) / total_samples
            precision_per_class.append(precision * weight)
        else:
            precision_per_class.append(precision)
    
    return np.sum(precision_per_class) if average == 'weighted' else np.mean(precision_per_class)

def recall_score(y_true, y_pred, average='weighted'):
    """Вычисление полноты (recall) для многоклассовой классификации."""
    classes = np.unique(y_true)
    recall_per_class = []
    total_samples = len(y_true)
    
    for cls in classes:
        true_positives = np.sum((y_pred == cls) & (y_true == cls))
        actual_positives = np.sum(y_true == cls)
        
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives
        
        if average == 'weighted':
            weight = np.sum(y_true == cls) / total_samples
            recall_per_class.append(recall * weight)
        else:
            recall_per_class.append(recall)
    
    return np.sum(recall_per_class) if average == 'weighted' else np.mean(recall_per_class)