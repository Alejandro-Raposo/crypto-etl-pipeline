"""
Módulo para Ensemble Methods (combinación de múltiples modelos).

Combina Random Forest y Naive Bayes usando Voting Classifier
para mejorar robustez de predicciones.

Siguiendo arquitectura estricta:
- Funciones pequeñas (<50 líneas)
- Documentación completa
- Logging en lugar de prints
"""

import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)


def train_ensemble_model(X_train, y_train, voting='soft'):
    """
    Entrena modelo ensemble combinando Random Forest y Naive Bayes.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        voting: Tipo de votación ('soft' para probabilidades, 'hard' para mayoría)
    
    Returns:
        Dict con:
            - model: Modelo ensemble entrenado
            - estimators: Lista de estimadores individuales
            - voting: Tipo de votación usado
    """
    logger.info("Creando modelo ensemble...")
    logger.info(f"  Estimadores: Random Forest (100 trees) + Naive Bayes")
    logger.info(f"  Voting: {voting}")
    
    # Definir estimadores
    estimators = [
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )),
        ('naive_bayes', GaussianNB())
    ]
    
    # Crear ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    
    logger.info("Entrenando modelo ensemble...")
    ensemble.fit(X_train, y_train)
    logger.info("Modelo ensemble entrenado exitosamente")
    
    return {
        'model': ensemble,
        'estimators': [name for name, _ in estimators],
        'voting': voting
    }


def compare_ensemble_vs_individual(X_train, X_test, y_train, y_test):
    """
    Compara rendimiento del ensemble vs modelos individuales.
    
    Args:
        X_train, X_test: Features de train/test
        y_train, y_test: Targets de train/test
    
    Returns:
        Dict con accuracy de cada modelo
    """
    logger.info("Comparando ensemble vs modelos individuales...")
    
    # Random Forest solo
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)
    
    # Naive Bayes solo
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_accuracy = nb.score(X_test, y_test)
    
    # Ensemble
    ensemble_result = train_ensemble_model(X_train, y_train)
    ensemble_accuracy = ensemble_result['model'].score(X_test, y_test)
    
    logger.info(f"  Random Forest: {rf_accuracy:.2%}")
    logger.info(f"  Naive Bayes: {nb_accuracy:.2%}")
    logger.info(f"  Ensemble: {ensemble_accuracy:.2%}")
    
    return {
        'random_forest_accuracy': float(rf_accuracy),
        'naive_bayes_accuracy': float(nb_accuracy),
        'ensemble_accuracy': float(ensemble_accuracy),
        'ensemble_improvement': float(ensemble_accuracy - max(rf_accuracy, nb_accuracy))
    }

