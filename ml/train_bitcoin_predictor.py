import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import train_model, split_train_test, check_class_balance, save_model
from ml.model_evaluator import generate_evaluation_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_bitcoin_price_predictor(crypto_id='bitcoin', days=7):
    """
    Entrena modelo completo de predicción de precio de Bitcoin.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia a usar (default: 7)
    
    Returns:
        Dict con modelo y métricas de evaluación
    """
    logger.info(f"Iniciando entrenamiento para {crypto_id}")
    logger.info(f"Usando {days} días de datos históricos")
    
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    logger.info(f"Datos cargados: {len(df)} registros")
    
    df_with_target = create_target(df)
    logger.info(f"Target creado: {len(df_with_target)} registros válidos")
    
    feature_cols, target_col = select_features(df_with_target)
    logger.info(f"Features seleccionadas: {len(feature_cols)}")
    
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    
    balance = check_class_balance(y)
    logger.info(f"Balance de clases: {balance}")
    
    X_normalized = normalize_features(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    logger.info("Entrenando modelo Random Forest...")
    model = train_model(X_train, y_train, n_estimators=100)
    logger.info("Modelo entrenado exitosamente")
    
    y_pred = model.predict(X_test)
    
    evaluation = generate_evaluation_report(y_test, y_pred)
    logger.info("Evaluación del modelo:")
    logger.info(f"  Accuracy:  {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall:    {evaluation['recall']:.4f}")
    logger.info(f"  F1-Score:  {evaluation['f1_score']:.4f}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_model(model, f"{crypto_id}_predictor_{timestamp}")
    logger.info(f"Modelo guardado en: {model_path}")
    
    return {
        'model': model,
        'model_path': model_path,
        'evaluation': evaluation,
        'feature_cols': feature_cols,
        'balance': balance,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("BITCOIN PRICE DIRECTION PREDICTOR")
    logger.info("="*60)
    
    result = train_bitcoin_price_predictor(crypto_id='bitcoin', days=7)
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*60)
    logger.info(f"\nModelo guardado: {result['model_path']}")
    logger.info(f"\nMétricas finales:")
    logger.info(f"  Accuracy:  {result['evaluation']['accuracy']:.2%}")
    logger.info(f"  Precision: {result['evaluation']['precision']:.2%}")
    logger.info(f"  Recall:    {result['evaluation']['recall']:.2%}")
    logger.info(f"  F1-Score:  {result['evaluation']['f1_score']:.2%}")

