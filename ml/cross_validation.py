from sklearn.model_selection import cross_val_score

def evaluate_with_cross_validation(X, y, model, cv=5):
    """
    Evalúa modelo con validación cruzada.
    
    Args:
        X: Features normalizadas (DataFrame o array)
        y: Target (Series o array)
        model: Modelo scikit-learn a evaluar
        cv: Número de folds (default 5)
    
    Returns:
        Dict con scores, mean, std, min, max
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'scores': scores.tolist(),
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max())
    }

