import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer_optimized import create_target, select_features_optimized, normalize_features
from ml.model_trainer import train_naive_bayes, split_train_test, check_class_balance, save_model
from ml.model_evaluator import generate_evaluation_report
from ml.cross_validation import evaluate_with_cross_validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_optimized_model(crypto_id='bitcoin', days=30):
    """
    Entrena modelo con top 12 features optimizadas.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia a usar (default: 30)
    
    Returns:
        Dict con modelo y métricas de evaluación
    """
    logger.info(f"Iniciando entrenamiento OPTIMIZADO para {crypto_id}")
    logger.info(f"Usando {days} días de datos históricos")
    
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    logger.info(f"Datos cargados: {len(df)} registros")
    
    df_with_target = create_target(df)
    logger.info(f"Target creado: {len(df_with_target)} registros válidos")
    
    feature_cols, target_col = select_features_optimized(df_with_target)
    logger.info(f"Features OPTIMIZADAS seleccionadas: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")
    
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    
    balance = check_class_balance(y)
    logger.info(f"Balance de clases: {balance}")
    
    X_normalized = normalize_features(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    logger.info("Entrenando modelo Naive Bayes...")
    model = train_naive_bayes(X_train, y_train)
    logger.info("Modelo entrenado exitosamente")
    
    y_pred = model.predict(X_test)
    
    evaluation = generate_evaluation_report(y_test, y_pred)
    logger.info("Evaluación del modelo:")
    logger.info(f"  Test Accuracy:  {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall:    {evaluation['recall']:.4f}")
    logger.info(f"  F1-Score:  {evaluation['f1_score']:.4f}")
    
    logger.info("\nEvaluando con Cross-Validation...")
    cv_result = evaluate_with_cross_validation(X_normalized, y, model, cv=5)
    logger.info(f"  CV Accuracy: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_model(model, f"{crypto_id}_optimized_{timestamp}")
    logger.info(f"Modelo guardado en: {model_path}")
    
    return {
        'model': model,
        'model_path': model_path,
        'evaluation': evaluation,
        'cv_result': cv_result,
        'feature_cols': feature_cols,
        'balance': balance,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("MODELO OPTIMIZADO (TOP 12 FEATURES)")
    logger.info("="*60)
    
    result = train_optimized_model(crypto_id='bitcoin', days=30)
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*60)
    logger.info(f"\nModelo guardado: {result['model_path']}")
    logger.info(f"\nMétricas finales:")
    logger.info(f"  Test Accuracy:  {result['evaluation']['accuracy']:.2%}")
    logger.info(f"  CV Accuracy:    {result['cv_result']['mean']:.2%} ± {result['cv_result']['std']:.2%}")
    logger.info(f"  Precision: {result['evaluation']['precision']:.2%}")
    logger.info(f"  Recall:    {result['evaluation']['recall']:.2%}")
    logger.info(f"  F1-Score:  {result['evaluation']['f1_score']:.2%}")

