import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timezone

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("./data/raw")
PROC_DIR = Path("./data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_raw():
    """
    Carga el último archivo JSON de la carpeta raw.
    Maneja la nueva estructura con metadatos.
    """
    files = sorted(RAW_DIR.glob("coins_*.json")) 
    if not files:
        raise FileNotFoundError("No raw files found in data/raw")
    
    latest_file = files[-1]
    logger.info(f"Loading raw data from: {latest_file}")
    
    raw_data = json.loads(latest_file.read_text(encoding="utf-8"))
    
    # Manejar nueva estructura con metadatos
    if isinstance(raw_data, dict) and "data" in raw_data:
        logger.info(f"Found structured data with metadata")
        return raw_data["data"], raw_data.get("metadata", {})
    else:
        # Compatibilidad con formato anterior
        logger.info("Found legacy data format")
        return raw_data, {}  

def calculate_ml_features(df):
    """
    Calcula features adicionales para machine learning.
    """
    logger.info("Calculating ML features...")
    
    # Características de precio
    df['price_log'] = np.log1p(df['current_price'])
    df['market_cap_log'] = np.log1p(df['market_cap'])
    df['volume_log'] = np.log1p(df['total_volume'])
    
    # Ratios importantes
    df['volume_to_market_cap_ratio'] = df['total_volume'] / df['market_cap']
    df['volume_millions'] = df['total_volume'] / 1e6
    
    # Categorización por market cap
    df['market_cap_category'] = pd.cut(
        df['market_cap_millions'], 
        bins=[0, 100, 1000, 10000, np.inf],
        labels=['micro', 'small', 'mid', 'large']
    )
    
    # Volatilidad features (cuando tengamos datos históricos)
    if 'price_change_24h' in df.columns:
        df['price_volatility_24h'] = abs(df['price_change_24h'])
    elif 'price_change_percentage_24h' in df.columns:
        df['price_volatility_24h'] = abs(df['price_change_percentage_24h'])
    
    # Rankings
    df['market_cap_rank_normalized'] = df['market_cap_rank'] / df['market_cap_rank'].max()
    
    return df

def transform(raw_data, metadata=None):
    """
    Limpieza y transformación del JSON a DataFrame listo para cargar.
    Incluye feature engineering para ML.
    """
    logger.info("Starting data transformation...")
    
    df = pd.DataFrame(raw_data)
    logger.info(f"Initial dataframe shape: {df.shape}")
    
    # Seleccionar columnas base + nuevas características para ML
    base_columns = [
        'id', 'symbol', 'name', 'current_price', 'market_cap', 'total_volume',
        'market_cap_rank', 'price_change_percentage_24h', 'last_updated'
    ]
    
    # Añadir nuevas columnas de cambio de precio si están disponibles
    ml_columns = [
        'price_change_percentage_1h_in_currency',
        'price_change_percentage_7d_in_currency',
        'high_24h', 'low_24h'
    ]
    
    available_columns = [col for col in base_columns + ml_columns if col in df.columns]
    df = df[available_columns]
    
    logger.info(f"Selected {len(available_columns)} columns for transformation")
    
    # Limpiar datos
    df = df.dropna(subset=['current_price'])
    
    # Conversiones de fecha
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['date'] = df['last_updated'].dt.date
    df['hour'] = df['last_updated'].dt.hour
    
    # Renombrar columnas para claridad
    column_renames = {
        'price_change_percentage_1h_in_currency': 'price_change_1h',
        'price_change_percentage_24h': 'price_change_24h',
        'price_change_percentage_7d_in_currency': 'price_change_7d'
    }
    df = df.rename(columns=column_renames)
    
    # Calcular market cap en millones
    df['market_cap_millions'] = df['market_cap'] / 1e6
    
    # Calcular features adicionales para ML
    df = calculate_ml_features(df)
    
    # Añadir metadatos de extracción si están disponibles
    if metadata:
        df['extraction_timestamp'] = metadata.get('extraction_timestamp')
        df['pipeline_version'] = metadata.get('pipeline_version', 'unknown')
    
    logger.info(f"Final dataframe shape: {df.shape}")
    logger.info(f"Features engineered: {df.shape[1] - len(available_columns)} new features")
    
    return df

def main():
    """
    Función principal mejorada para ML pipeline.
    """
    try:
        logger.info("Starting data transformation for ML pipeline")
        
        # Cargar datos raw con metadatos
        raw_data, metadata = load_latest_raw()
        
        # Transformar datos con feature engineering
        df = transform(raw_data, metadata)
        
        # Generar archivo de salida
        now_utc = datetime.now(timezone.utc)
        out_file = PROC_DIR / f"coins_processed_{now_utc.strftime('%Y%m%d_%H%M%S')}.parquet"
        
        # Guardar en formato Parquet (optimizado para ML)
        df.to_parquet(out_file, index=False)
        
        logger.info(f"✅ Processed data saved: {out_file}")
        logger.info(f"📊 Final dataset: {df.shape[0]} rows, {df.shape[1]} features")
        logger.info(f"🔧 ML features included: price_log, volume_ratios, volatility, categories")
        
        return out_file
        
    except Exception as e:
        logger.error(f"❌ Transformation failed: {e}")
        raise

if __name__ == "__main__":
    main()
