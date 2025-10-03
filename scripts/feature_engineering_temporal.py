import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from google.cloud import bigquery
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

PROJECT = "crypto-etl-proyect"
DATASET = "crypto_dataset"
TABLE_HISTORICAL = "prices_historical"
TABLE_ML_FEATURES = "prices_ml_features"

credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(f"Credenciales no encontradas: {credentials_path}")

def load_historical_data(client):
    query = f"""
    SELECT 
        id,
        symbol,
        name,
        current_price,
        market_cap,
        total_volume,
        last_updated,
        partition_date,
        price_change_24h,
        high_24h,
        low_24h
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    ORDER BY id, last_updated
    """
    logger.info("Cargando datos historicos de BigQuery")
    df = client.query(query).to_dataframe()
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df = df.sort_values(['id', 'last_updated'])
    logger.info(f"Datos cargados: {len(df)} registros, {df['id'].nunique()} cryptos")
    return df

def calculate_lag_features(df):
    logger.info("Calculando lag features")
    df['price_lag_1h'] = df.groupby('id')['current_price'].shift(1)
    df['price_lag_3h'] = df.groupby('id')['current_price'].shift(3)
    df['price_lag_6h'] = df.groupby('id')['current_price'].shift(6)
    df['price_lag_12h'] = df.groupby('id')['current_price'].shift(12)
    df['price_lag_24h'] = df.groupby('id')['current_price'].shift(24)
    df['price_lag_48h'] = df.groupby('id')['current_price'].shift(48)
    df['price_lag_7d'] = df.groupby('id')['current_price'].shift(24 * 7)
    df['volume_lag_1h'] = df.groupby('id')['total_volume'].shift(1)
    df['volume_lag_24h'] = df.groupby('id')['total_volume'].shift(24)
    return df

def calculate_price_changes(df):
    logger.info("Calculando cambios de precio")
    df['price_change_pct_1h'] = ((df['current_price'] - df['price_lag_1h']) / df['price_lag_1h']) * 100
    df['price_change_pct_3h'] = ((df['current_price'] - df['price_lag_3h']) / df['price_lag_3h']) * 100
    df['price_change_pct_6h'] = ((df['current_price'] - df['price_lag_6h']) / df['price_lag_6h']) * 100
    df['price_change_pct_12h'] = ((df['current_price'] - df['price_lag_12h']) / df['price_lag_12h']) * 100
    df['price_change_pct_24h'] = ((df['current_price'] - df['price_lag_24h']) / df['price_lag_24h']) * 100
    df['price_change_pct_48h'] = ((df['current_price'] - df['price_lag_48h']) / df['price_lag_48h']) * 100
    df['price_change_pct_7d'] = ((df['current_price'] - df['price_lag_7d']) / df['price_lag_7d']) * 100
    return df

def calculate_rolling_features(df):
    logger.info("Calculando rolling windows")
    for window in [6, 12, 24, 48, 168]:
        df[f'price_ma_{window}h'] = df.groupby('id')['current_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'price_std_{window}h'] = df.groupby('id')['current_price'].transform(
            lambda x: x.rolling(window=window, min_periods=2).std()
        )
        df[f'price_min_{window}h'] = df.groupby('id')['current_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df[f'price_max_{window}h'] = df.groupby('id')['current_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        df[f'volume_ma_{window}h'] = df.groupby('id')['total_volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    return df

def calculate_momentum_indicators(df):
    logger.info("Calculando indicadores de momentum")
    df['price_momentum_6h'] = df.groupby('id')['current_price'].transform(lambda x: x.diff(6))
    df['price_momentum_12h'] = df.groupby('id')['current_price'].transform(lambda x: x.diff(12))
    df['price_momentum_24h'] = df.groupby('id')['current_price'].transform(lambda x: x.diff(24))
    df['volume_change_24h'] = df['total_volume'] - df['volume_lag_24h']
    df['volume_change_pct_24h'] = ((df['total_volume'] - df['volume_lag_24h']) / df['volume_lag_24h']) * 100
    return df

def calculate_volatility_features(df):
    logger.info("Calculando features de volatilidad")
    for window in [24, 48, 168]:
        df[f'volatility_{window}h'] = df.groupby('id')['current_price'].transform(
            lambda x: x.pct_change().rolling(window=window, min_periods=2).std() * 100
        )
    df['price_range_24h'] = ((df['high_24h'] - df['low_24h']) / df['low_24h']) * 100
    return df

def calculate_rsi_components(df):
    logger.info("Calculando componentes RSI")
    df['price_change'] = df.groupby('id')['current_price'].diff()
    df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
    for window in [14, 24]:
        df[f'avg_gain_{window}h'] = df.groupby('id')['gain'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'avg_loss_{window}h'] = df.groupby('id')['loss'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rs_{window}h'] = df[f'avg_gain_{window}h'] / (df[f'avg_loss_{window}h'] + 1e-10)
        df[f'rsi_{window}h'] = 100 - (100 / (1 + df[f'rs_{window}h']))
    return df

def calculate_relative_features(df):
    logger.info("Calculando features relativas")
    for window in [24, 48, 168]:
        df[f'price_vs_ma_{window}h'] = ((df['current_price'] - df[f'price_ma_{window}h']) / df[f'price_ma_{window}h']) * 100
        df[f'price_normalized_{window}h'] = (df['current_price'] - df[f'price_min_{window}h']) / (df[f'price_max_{window}h'] - df[f'price_min_{window}h'] + 1e-10)
    return df

def add_time_features(df):
    logger.info("Agregando features temporales")
    df['hour'] = df['last_updated'].dt.hour
    df['day_of_week'] = df['last_updated'].dt.dayofweek
    df['day_of_month'] = df['last_updated'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

def validate_data_quality(df):
    logger.info("Validando calidad de datos")
    initial_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_counts = df[numeric_cols].isna().sum()
    nan_pct = (nan_counts / len(df)) * 100
    logger.info(f"Columnas con >50% NaN: {list(nan_pct[nan_pct > 50].index)}")
    logger.info(f"Registros finales: {len(df)} (eliminados: {initial_count - len(df)})")
    return df

def save_to_bigquery(client, df):
    table_ref = client.dataset(DATASET).table(TABLE_ML_FEATURES)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    logger.info(f"Guardando {len(df)} registros con {len(df.columns)} features en BigQuery")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    logger.info(f"Tabla {TABLE_ML_FEATURES} actualizada correctamente")

def generate_temporal_features():
    client = bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)
    logger.info("Iniciando feature engineering temporal")
    df = load_historical_data(client)
    if len(df) < 24:
        logger.warning(f"Datos insuficientes: {len(df)} registros. Se requieren minimo 24h de datos")
        return
    df = calculate_lag_features(df)
    df = calculate_price_changes(df)
    df = calculate_rolling_features(df)
    df = calculate_momentum_indicators(df)
    df = calculate_volatility_features(df)
    df = calculate_rsi_components(df)
    df = calculate_relative_features(df)
    df = add_time_features(df)
    df = validate_data_quality(df)
    logger.info(f"Features generadas: {len(df.columns)} columnas totales")
    logger.info(f"Top 10 features: {df.columns[-10:].tolist()}")
    save_to_bigquery(client, df)
    logger.info("Feature engineering temporal completado")

if __name__ == "__main__":
    generate_temporal_features()

