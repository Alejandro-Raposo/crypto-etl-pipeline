import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import split_train_test, train_model, train_naive_bayes
from ml.model_evaluator import generate_evaluation_report
from ml.cross_validation import evaluate_with_cross_validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_models(crypto_id='bitcoin', days=30):
    """
    Compara Random Forest vs Naive Bayes con mismos datos.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia (default: 30)
    
    Returns:
        Dict con comparación detallada
    """
    logger.info(f"Cargando datos de {crypto_id}...")
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    
    df_with_target = create_target(df)
    feature_cols, target_col = select_features(df_with_target)
    
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    X_normalized = normalize_features(X)
    
    logger.info(f"Datos: {len(X)} registros, {len(feature_cols)} features")
    
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO RANDOM FOREST")
    logger.info("="*60)
    rf_model = train_model(X_train, y_train, n_estimators=100)
    rf_pred = rf_model.predict(X_test)
    rf_eval = generate_evaluation_report(y_test, rf_pred)
    
    logger.info(f"Accuracy:  {rf_eval['accuracy']:.4f}")
    logger.info(f"Precision: {rf_eval['precision']:.4f}")
    logger.info(f"Recall:    {rf_eval['recall']:.4f}")
    logger.info(f"F1-Score:  {rf_eval['f1_score']:.4f}")
    
    rf_cv = evaluate_with_cross_validation(X_normalized, y, rf_model, cv=5)
    logger.info(f"CV Accuracy: {rf_cv['mean']:.4f} ± {rf_cv['std']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO NAIVE BAYES")
    logger.info("="*60)
    nb_model = train_naive_bayes(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_eval = generate_evaluation_report(y_test, nb_pred)
    
    logger.info(f"Accuracy:  {nb_eval['accuracy']:.4f}")
    logger.info(f"Precision: {nb_eval['precision']:.4f}")
    logger.info(f"Recall:    {nb_eval['recall']:.4f}")
    logger.info(f"F1-Score:  {nb_eval['f1_score']:.4f}")
    
    nb_cv = evaluate_with_cross_validation(X_normalized, y, nb_model, cv=5)
    logger.info(f"CV Accuracy: {nb_cv['mean']:.4f} ± {nb_cv['std']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("COMPARACIÓN FINAL")
    logger.info("="*60)
    
    logger.info(f"\nRandom Forest:")
    logger.info(f"  Test Accuracy:  {rf_eval['accuracy']:.2%}")
    logger.info(f"  CV Accuracy:    {rf_cv['mean']:.2%} ± {rf_cv['std']:.2%}")
    logger.info(f"  F1-Score:       {rf_eval['f1_score']:.2%}")
    
    logger.info(f"\nNaive Bayes:")
    logger.info(f"  Test Accuracy:  {nb_eval['accuracy']:.2%}")
    logger.info(f"  CV Accuracy:    {nb_cv['mean']:.2%} ± {nb_cv['std']:.2%}")
    logger.info(f"  F1-Score:       {nb_eval['f1_score']:.2%}")
    
    if nb_cv['mean'] > rf_cv['mean']:
        winner = "Naive Bayes"
        diff = (nb_cv['mean'] - rf_cv['mean']) * 100
    else:
        winner = "Random Forest"
        diff = (rf_cv['mean'] - nb_cv['mean']) * 100
    
    logger.info(f"\n🏆 GANADOR: {winner} (+{diff:.2f}% accuracy)")
    
    return {
        'random_forest': {'test': rf_eval, 'cv': rf_cv},
        'naive_bayes': {'test': nb_eval, 'cv': nb_cv},
        'winner': winner
    }

if __name__ == '__main__':
    compare_models(crypto_id='bitcoin', days=30)

