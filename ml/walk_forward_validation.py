"""
Módulo para Walk-Forward Validation (validación temporal deslizante).

Esta técnica es más realista para series temporales que cross-validation
tradicional, ya que respeta el orden temporal estricto.

Siguiendo arquitectura estricta:
- Funciones pequeñas (<50 líneas)
- Documentación completa
- Logging en lugar de prints
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def walk_forward_validation(df, feature_cols, target_col, train_window=60, test_window=10, step=5):
    """
    Realiza validación temporal deslizante.
    
    En cada iteración:
    - Entrena con los últimos train_window registros
    - Testea con los siguientes test_window registros
    - Avanza step registros y repite
    
    Args:
        df: DataFrame ordenado temporalmente
        feature_cols: Lista de nombres de features
        target_col: Nombre de columna target
        train_window: Tamaño de ventana de entrenamiento (default 60)
        test_window: Tamaño de ventana de test (default 10)
        step: Cuántos registros avanzar en cada iteración (default 5)
    
    Returns:
        Dict con:
            - fold_scores: Lista de accuracy por cada fold
            - mean_accuracy: Accuracy promedio
            - std_accuracy: Desviación estándar
            - n_folds: Número de folds ejecutados
    """
    logger.info(f"Iniciando Walk-Forward Validation...")
    logger.info(f"  Train window: {train_window}")
    logger.info(f"  Test window: {test_window}")
    logger.info(f"  Step: {step}")
    
    fold_scores = []
    fold_num = 0
    
    # Iterar con ventana deslizante
    start = 0
    while start + train_window + test_window <= len(df):
        fold_num += 1
        
        # Definir índices de train y test
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window
        
        # Extraer datos
        X_train = df.iloc[train_start:train_end][feature_cols]
        y_train = df.iloc[train_start:train_end][target_col]
        X_test = df.iloc[test_start:test_end][feature_cols]
        y_test = df.iloc[test_start:test_end][target_col]
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).sum() / len(y_test)
        
        fold_scores.append(float(accuracy))
        
        logger.info(f"  Fold {fold_num}: Accuracy = {accuracy:.2%}")
        
        # Avanzar ventana
        start += step
    
    # Calcular métricas agregadas
    mean_accuracy = float(np.mean(fold_scores))
    std_accuracy = float(np.std(fold_scores))
    
    logger.info(f"\nWalk-Forward Validation completado:")
    logger.info(f"  Folds ejecutados: {fold_num}")
    logger.info(f"  Accuracy promedio: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
    
    return {
        'fold_scores': fold_scores,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'n_folds': fold_num
    }

