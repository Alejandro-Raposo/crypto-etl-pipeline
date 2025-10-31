import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import split_train_test, train_model
from ml.feature_importance import analyze_feature_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_features(crypto_id='bitcoin', days=30):
    """
    Analiza importancia de features usando Random Forest.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia (default: 30)
    
    Returns:
        DataFrame con feature importance
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
    
    logger.info("\nEntrenando Random Forest para análisis...")
    model = train_model(X_train, y_train, n_estimators=100)
    
    logger.info("\nAnalizando importancia de features...")
    importance_df = analyze_feature_importance(model, feature_cols)
    
    logger.info("\n" + "="*60)
    logger.info("TOP 10 FEATURES MÁS IMPORTANTES")
    logger.info("="*60)
    
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"{row['rank']:2d}. {row['feature']:25s} - {row['importance_pct']:6.2f}%")
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN")
    logger.info("="*60)
    
    top_3_importance = importance_df.head(3)['importance_pct'].sum()
    top_5_importance = importance_df.head(5)['importance_pct'].sum()
    
    logger.info(f"Top 3 features concentran: {top_3_importance:.1f}% de importancia")
    logger.info(f"Top 5 features concentran: {top_5_importance:.1f}% de importancia")
    
    return importance_df

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("ANÁLISIS DE FEATURE IMPORTANCE")
    logger.info("="*60)
    
    importance_df = analyze_features(crypto_id='bitcoin', days=30)
    
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS COMPLETADO")
    logger.info("="*60)
    logger.info(f"\nDataFrame guardado con {len(importance_df)} features analizadas")

