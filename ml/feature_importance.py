import pandas as pd

def analyze_feature_importance(model, feature_names):
    """
    Analiza importancia de features en modelos tree-based.
    
    Args:
        model: Modelo tree-based entrenado (RF, XGBoost, etc.)
        feature_names: Lista de nombres de features
    
    Returns:
        DataFrame ordenado por importancia descendente
    
    Raises:
        ValueError: Si el modelo no soporta feature_importances_
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Modelo no soporta feature_importances_. Use modelos tree-based (RandomForest, XGBoost, etc.)")
    
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    })
    
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100
    
    return df

