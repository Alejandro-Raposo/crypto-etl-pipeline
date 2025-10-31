"""
Módulo para backtesting temporal de predicciones.

Simula predicciones en datos históricos para evaluar rendimiento real,
evitando data leakage temporal.

Siguiendo arquitectura estricta:
- Funciones pequeñas (<50 líneas)
- Documentación completa
- Logging en lugar de prints
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def backtest_predictions(model, df, feature_cols, target_col, window=24):
    """
    Simula predicciones pasadas para evaluar rendimiento real del modelo.
    
    Cada predicción solo usa datos hasta ese momento (sin data leakage).
    
    Args:
        model: Modelo entrenado (debe tener métodos predict y predict_proba)
        df: DataFrame con datos históricos ordenados temporalmente
        feature_cols: Lista de nombres de columnas de features
        target_col: Nombre de la columna target
        window: Número de registros iniciales a omitir (default 24)
    
    Returns:
        Dict con:
            - results: DataFrame con predicciones vs realidad
            - accuracy: Accuracy global del backtesting
            - total_predictions: Total de predicciones realizadas
            - correct_predictions: Número de predicciones correctas
    """
    logger.info(f"Iniciando backtesting con ventana de {window} registros...")
    
    results = []
    
    # Iterar desde window hasta el final
    for i in range(window, len(df)):
        # Usar solo el registro actual para predecir
        X = df.iloc[i:i+1][feature_cols]
        y_true = df.iloc[i][target_col]
        
        # Realizar predicción
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0]
        
        results.append({
            'timestamp': df.iloc[i]['last_updated'],
            'predicted': int(y_pred),
            'actual': int(y_true),
            'correct': bool(y_pred == y_true),
            'probability': float(y_proba[y_pred]),
            'price': df.iloc[i]['current_price']
        })
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Calcular métricas
    accuracy = results_df['correct'].sum() / len(results_df)
    correct_predictions = int(results_df['correct'].sum())
    total_predictions = len(results_df)
    
    logger.info(f"Backtesting completado:")
    logger.info(f"  Total predicciones: {total_predictions}")
    logger.info(f"  Predicciones correctas: {correct_predictions}")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    
    return {
        'results': results_df,
        'accuracy': float(accuracy),
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions
    }

