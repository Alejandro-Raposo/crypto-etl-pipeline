import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.cross_validation import evaluate_with_cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_models_with_cv(crypto_id='bitcoin', days=30, cv=5):
    """
    Evalúa múltiples modelos con cross-validation.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia (default: 30)
        cv: Número de folds (default: 5)
    
    Returns:
        Dict con resultados de CV para cada modelo
    """
    logger.info(f"Cargando datos de {crypto_id} ({days} días)...")
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    
    df_with_target = create_target(df)
    feature_cols, target_col = select_features(df_with_target)
    
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    X_normalized = normalize_features(X)
    
    logger.info(f"Datos preparados: {len(X)} registros, {len(feature_cols)} features")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=20,
            random_state=42
        ),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nEvaluando {name} con {cv}-fold CV...")
        cv_result = evaluate_with_cross_validation(X_normalized, y, model, cv=cv)
        results[name] = cv_result
        
        mean_acc = cv_result['mean']
        std_acc = cv_result['std']
        ci_lower = mean_acc - 1.96 * std_acc
        ci_upper = mean_acc + 1.96 * std_acc
        
        logger.info(f"  Accuracy media: {mean_acc:.4f}")
        logger.info(f"  Desviación std: {std_acc:.4f}")
        logger.info(f"  Intervalo 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  Min: {cv_result['min']:.4f} | Max: {cv_result['max']:.4f}")
    
    return results

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("EVALUACIÓN CON CROSS-VALIDATION")
    logger.info("="*60)
    
    results = evaluate_models_with_cv(crypto_id='bitcoin', days=30, cv=5)
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*60)
    
    for model_name, cv_result in results.items():
        mean_acc = cv_result['mean']
        std_acc = cv_result['std']
        logger.info(f"\n{model_name}:")
        logger.info(f"  CV Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")

