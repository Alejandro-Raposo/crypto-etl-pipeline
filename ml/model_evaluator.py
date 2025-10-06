import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calculate_accuracy(y_true, y_pred):
    """
    Calcula accuracy del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Accuracy entre 0 y 1
    """
    return float(accuracy_score(y_true, y_pred))

def calculate_precision(y_true, y_pred):
    """
    Calcula precision del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Precision entre 0 y 1
    """
    return float(precision_score(y_true, y_pred, zero_division=0))

def calculate_recall(y_true, y_pred):
    """
    Calcula recall del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Recall entre 0 y 1
    """
    return float(recall_score(y_true, y_pred, zero_division=0))

def calculate_f1_score(y_true, y_pred):
    """
    Calcula F1-Score del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        F1-Score entre 0 y 1
    """
    return float(f1_score(y_true, y_pred, zero_division=0))

def generate_confusion_matrix(y_true, y_pred):
    """
    Genera matriz de confusión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Matriz de confusión como numpy array
    """
    return confusion_matrix(y_true, y_pred)

def generate_evaluation_report(y_true, y_pred):
    """
    Genera reporte completo de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Dict con todas las métricas
    """
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1_score(y_true, y_pred),
        'confusion_matrix': generate_confusion_matrix(y_true, y_pred).tolist()
    }

