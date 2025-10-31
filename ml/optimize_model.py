"""
Script para ejecutar todas las optimizaciones del modelo ML.

Implementa:
1. Grid Search para optimizar hiperparámetros
2. Backtesting temporal
3. Walk-Forward Validation
4. Ensemble Methods

Siguiendo arquitectura estricta del proyecto.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import split_train_test, check_class_balance, save_model, train_naive_bayes
from ml.model_evaluator import generate_evaluation_report
from ml.cross_validation import evaluate_with_cross_validation
from ml.hyperparameter_tuning import optimize_random_forest
from ml.backtesting import backtest_predictions
from ml.walk_forward_validation import walk_forward_validation
from ml.ensemble_predictor import train_ensemble_model, compare_ensemble_vs_individual

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize_bitcoin_model(crypto_id='bitcoin', days=30):
    """
    Ejecuta todas las optimizaciones sobre el modelo de Bitcoin.
    
    Args:
        crypto_id: ID de la crypto (default: 'bitcoin')
        days: Días de historia a usar (default: 30)
    
    Returns:
        Dict con todos los resultados de optimización
    """
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN COMPLETA DEL MODELO ML")
    logger.info("="*70)
    
    # ============================
    # 1. CARGAR Y PREPARAR DATOS
    # ============================
    logger.info(f"\n1️⃣  Cargando datos de {crypto_id} ({days} días)...")
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    logger.info(f"   Datos cargados: {len(df)} registros")
    
    df_with_target = create_target(df)
    feature_cols, target_col = select_features(df_with_target)
    
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    
    balance = check_class_balance(y)
    logger.info(f"   Balance de clases: {balance}")
    
    X_normalized = normalize_features(X)
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    
    logger.info(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"   Features: {len(feature_cols)}")
    
    # ============================
    # 2. GRID SEARCH (Random Forest)
    # ============================
    logger.info("\n2️⃣  GRID SEARCH - Optimización de Hiperparámetros")
    logger.info("-"*70)
    
    grid_result = optimize_random_forest(X_train, y_train, cv=5)
    
    logger.info(f"   Mejores parámetros encontrados:")
    for param, value in grid_result['best_params'].items():
        logger.info(f"     • {param}: {value}")
    logger.info(f"   Mejor CV Score: {grid_result['best_score']:.2%}")
    
    # Evaluar en test
    rf_optimized = grid_result['best_model']
    y_pred_rf = rf_optimized.predict(X_test)
    rf_eval = generate_evaluation_report(y_test, y_pred_rf)
    
    logger.info(f"   Test Accuracy: {rf_eval['accuracy']:.2%}")
    logger.info(f"   F1-Score: {rf_eval['f1_score']:.2%}")
    
    # ============================
    # 3. BACKTESTING TEMPORAL
    # ============================
    logger.info("\n3️⃣  BACKTESTING TEMPORAL")
    logger.info("-"*70)
    
    # Preparar DataFrame para backtesting (necesita last_updated y current_price)
    df_backtest = df_with_target.copy()
    df_backtest_features = df_backtest[feature_cols + [target_col, 'last_updated', 'current_price']]
    
    # Usar modelo Naive Bayes (más ligero)
    nb_model = train_naive_bayes(X_train, y_train)
    
    backtest_result = backtest_predictions(
        nb_model,
        df_backtest_features,
        feature_cols,
        target_col,
        window=24
    )
    
    logger.info(f"   Backtesting Accuracy: {backtest_result['accuracy']:.2%}")
    logger.info(f"   Total predicciones: {backtest_result['total_predictions']}")
    logger.info(f"   Predicciones correctas: {backtest_result['correct_predictions']}")
    
    # ============================
    # 4. WALK-FORWARD VALIDATION
    # ============================
    logger.info("\n4️⃣  WALK-FORWARD VALIDATION")
    logger.info("-"*70)
    
    # Solo si tenemos suficientes datos (mínimo 70 registros)
    if len(df_backtest_features) >= 70:
        wf_result = walk_forward_validation(
            df_backtest_features,
            feature_cols,
            target_col,
            train_window=50,
            test_window=10,
            step=5
        )
        
        logger.info(f"   WF Accuracy promedio: {wf_result['mean_accuracy']:.2%} ± {wf_result['std_accuracy']:.2%}")
        logger.info(f"   Número de folds: {wf_result['n_folds']}")
        logger.info(f"   Rango: [{min(wf_result['fold_scores']):.2%}, {max(wf_result['fold_scores']):.2%}]")
    else:
        logger.warning(f"   Datos insuficientes para Walk-Forward ({len(df_backtest_features)} < 70)")
        wf_result = None
    
    # ============================
    # 5. ENSEMBLE METHODS
    # ============================
    logger.info("\n5️⃣  ENSEMBLE METHODS (RF + Naive Bayes)")
    logger.info("-"*70)
    
    ensemble_comparison = compare_ensemble_vs_individual(X_train, X_test, y_train, y_test)
    
    logger.info(f"   Random Forest: {ensemble_comparison['random_forest_accuracy']:.2%}")
    logger.info(f"   Naive Bayes: {ensemble_comparison['naive_bayes_accuracy']:.2%}")
    logger.info(f"   Ensemble: {ensemble_comparison['ensemble_accuracy']:.2%}")
    
    improvement = ensemble_comparison['ensemble_improvement']
    if improvement > 0:
        logger.info(f"   ✅ Mejora del ensemble: +{improvement:.2%}")
    else:
        logger.info(f"   ⚠️  Ensemble no supera mejor individual: {improvement:.2%}")
    
    # ============================
    # 6. RESUMEN COMPARATIVO
    # ============================
    logger.info("\n" + "="*70)
    logger.info("RESUMEN COMPARATIVO DE TODAS LAS TÉCNICAS")
    logger.info("="*70)
    
    logger.info(f"\n📊 Accuracy Comparativo:")
    logger.info(f"   Random Forest Optimizado (Grid Search): {rf_eval['accuracy']:.2%}")
    logger.info(f"   Naive Bayes (Backtesting):              {backtest_result['accuracy']:.2%}")
    if wf_result:
        logger.info(f"   Walk-Forward Validation:                {wf_result['mean_accuracy']:.2%}")
    logger.info(f"   Ensemble (RF + NB):                     {ensemble_comparison['ensemble_accuracy']:.2%}")
    
    logger.info(f"\n🎯 Mejor técnica:")
    best_accuracy = max(
        rf_eval['accuracy'],
        backtest_result['accuracy'],
        wf_result['mean_accuracy'] if wf_result else 0,
        ensemble_comparison['ensemble_accuracy']
    )
    
    if best_accuracy == rf_eval['accuracy']:
        logger.info(f"   ✅ Random Forest Optimizado: {best_accuracy:.2%}")
    elif best_accuracy == backtest_result['accuracy']:
        logger.info(f"   ✅ Naive Bayes (Backtesting): {best_accuracy:.2%}")
    elif wf_result and best_accuracy == wf_result['mean_accuracy']:
        logger.info(f"   ✅ Walk-Forward Validation: {best_accuracy:.2%}")
    else:
        logger.info(f"   ✅ Ensemble: {best_accuracy:.2%}")
    
    # ============================
    # 7. GUARDAR MEJOR MODELO
    # ============================
    logger.info("\n💾 Guardando modelo optimizado...")
    
    # Entrenar modelo final con mejores parámetros encontrados
    ensemble_result = train_ensemble_model(X_train, y_train)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_model(ensemble_result['model'], f"{crypto_id}_optimized_ensemble_{timestamp}")
    
    logger.info(f"   Modelo guardado: {model_path}")
    
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info("="*70)
    
    return {
        'grid_search': grid_result,
        'rf_evaluation': rf_eval,
        'backtesting': backtest_result,
        'walk_forward': wf_result,
        'ensemble_comparison': ensemble_comparison,
        'best_accuracy': best_accuracy,
        'model_path': model_path,
        'feature_cols': feature_cols,
        'balance': balance
    }


if __name__ == '__main__':
    logger.info("Iniciando proceso de optimización completo...")
    
    result = optimize_bitcoin_model(crypto_id='bitcoin', days=30)
    
    logger.info("\n✅ Proceso completado exitosamente")
    logger.info(f"\n📁 Modelo final guardado en: {result['model_path']}")
    logger.info(f"🎯 Mejor accuracy alcanzado: {result['best_accuracy']:.2%}")

