"""
Módulo para optimización de hiperparámetros usando Grid Search.

Siguiendo arquitectura estricta:
- Funciones pequeñas (<50 líneas)
- Documentación completa
- Logging en lugar de prints
"""

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def optimize_random_forest(X_train, y_train, cv=5):
    """
    Optimiza hiperparámetros de Random Forest usando Grid Search.
    
    Args:
        X_train: Features de entrenamiento (numpy array o DataFrame)
        y_train: Target de entrenamiento (numpy array o Series)
        cv: Número de folds para cross-validation (default 5)
    
    Returns:
        Dict con:
            - best_model: Modelo Random Forest optimizado
            - best_params: Mejores parámetros encontrados
            - best_score: Mejor score de CV
            - cv_results: Resultados detallados de CV
    """
    logger.info("Iniciando Grid Search para Random Forest...")
    
    # Parámetros a optimizar (reducidos para evitar overfitting con pocos datos)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15]
    }
    
    logger.info(f"Espacio de búsqueda: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinaciones")
    
    # Modelo base
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid Search con CV
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Ejecutando Grid Search...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Mejor score encontrado: {grid_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'cv_results': grid_search.cv_results_
    }

