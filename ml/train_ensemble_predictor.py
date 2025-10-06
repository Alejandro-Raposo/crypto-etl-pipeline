import sys
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import split_train_test, check_class_balance, save_model
from ml.model_evaluator import generate_evaluation_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ensemble_predictor(crypto_id='bitcoin', days=7):
    """
    Entrena un ensemble de múltiples algoritmos (voting).
    
    Returns:
        Dict con modelo y métricas
    """
    logger.info(f"Cargando datos de {crypto_id}...")
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    df_with_target = create_target(df)
    
    feature_cols, target_col = select_features(df_with_target)
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    X_normalized = normalize_features(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    
    logger.info("Creando Ensemble de 3 algoritmos...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ],
        voting='soft'
    )
    
    logger.info("Entrenando ensemble...")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    evaluation = generate_evaluation_report(y_test, y_pred)
    
    logger.info("Evaluación del ensemble:")
    logger.info(f"  Accuracy:  {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall:    {evaluation['recall']:.4f}")
    logger.info(f"  F1-Score:  {evaluation['f1_score']:.4f}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_model(ensemble, f"{crypto_id}_ensemble_{timestamp}")
    logger.info(f"Modelo guardado en: {model_path}")
    
    return {
        'model': ensemble,
        'model_path': model_path,
        'evaluation': evaluation
    }

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("ENSEMBLE PREDICTOR (RF + GB + LR)")
    logger.info("="*60)
    
    result = train_ensemble_predictor()
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Accuracy:  {result['evaluation']['accuracy']:.2%}")
    print(f"F1-Score:  {result['evaluation']['f1_score']:.2%}")

