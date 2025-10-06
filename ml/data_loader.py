import os
import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

PROJECT = "crypto-etl-proyect"
DATASET = "crypto_dataset"
TABLE = "prices_ml_features"

credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(f"Credenciales no encontradas")

def _get_client():
    """Retorna cliente de BigQuery configurado"""
    return bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)

def load_crypto_data(crypto_id, days=7):
    """
    Carga datos históricos de una criptomoneda desde BigQuery.
    
    Args:
        crypto_id: ID de la criptomoneda (ej: 'bitcoin')
        days: Número de días de historia a cargar
    
    Returns:
        DataFrame con datos históricos y features ML
    """
    if not crypto_id or crypto_id is None:
        raise ValueError("crypto_id no puede estar vacío")
    
    client = _get_client()
    
    query = f"""
    SELECT 
        id,
        last_updated,
        current_price,
        price_lag_1h,
        price_lag_3h,
        price_lag_6h,
        price_ma_6h,
        price_ma_12h,
        price_ma_24h,
        price_change_pct_1h,
        price_change_pct_3h,
        rsi_14h,
        rsi_24h,
        volatility_24h,
        price_momentum_6h
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    WHERE id = '{crypto_id}'
      AND partition_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
    ORDER BY last_updated ASC
    """
    
    df = client.query(query).to_dataframe()
    df = df.dropna(subset=['price_lag_1h', 'price_ma_6h', 'rsi_14h'])
    df = df.sort_values('last_updated').reset_index(drop=True)
    
    logger.info(f"Cargados {len(df)} registros para {crypto_id}")
    return df

