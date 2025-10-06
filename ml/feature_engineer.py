import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_target(df):
    """
    Crea variable target binaria (1=sube, 0=baja) basada en precio futuro.
    
    Args:
        df: DataFrame con columna 'current_price'
    
    Returns:
        DataFrame con columna 'target' (sin última fila)
    """
    df = df.copy()
    df = df.sort_values('last_updated').reset_index(drop=True)
    df['price_future_1h'] = df['current_price'].shift(-1)
    df['target'] = (df['price_future_1h'] > df['current_price']).astype(int)
    df = df[:-1]
    return df

def select_features(df):
    """
    Selecciona features relevantes para el modelo.
    
    Args:
        df: DataFrame con todas las columnas
    
    Returns:
        Tuple (lista de features, nombre de target)
    """
    feature_cols = [
        'price_lag_1h',
        'price_lag_3h',
        'price_lag_6h',
        'price_ma_6h',
        'price_ma_12h',
        'price_ma_24h',
        'price_change_pct_1h',
        'price_change_pct_3h',
        'rsi_14h',
        'rsi_24h',
        'volatility_24h',
        'price_momentum_6h'
    ]
    available_features = [col for col in feature_cols if col in df.columns]
    return available_features, 'target'

def normalize_features(features):
    """
    Normaliza features entre 0 y 1 usando MinMaxScaler.
    
    Args:
        features: DataFrame con features numéricas
    
    Returns:
        DataFrame normalizado
    """
    features_clean = features.fillna(features.mean())
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(features_clean),
        columns=features.columns,
        index=features.index
    )
    return normalized

