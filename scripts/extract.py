import requests  
import json     
import logging
from datetime import datetime, timezone
from pathlib import Path
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("./data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)  

def fetch_coingecko_top(n=100, vs_currency="usd", max_retries=3):
    """
    Llama a la API de CoinGecko y devuelve los n principales criptos en la moneda vs_currency.
    Incluye retry logic y mejor manejo de errores para ML pipeline.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d"  # Más datos para ML
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching data from CoinGecko (attempt {attempt + 1}/{max_retries})")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            
            data = r.json()
            logger.info(f"Successfully fetched {len(data)} coins")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("All retry attempts failed")
                raise 

def add_extraction_metadata(data):
    """
    Añade metadatos importantes para el pipeline de ML.
    """
    extraction_time = datetime.now(timezone.utc)
    metadata = {
        "extraction_timestamp": extraction_time.isoformat(),
        "extraction_timestamp_unix": extraction_time.timestamp(),
        "total_coins": len(data),
        "data_source": "coingecko_markets_api",
        "pipeline_version": "v1.1_ml_ready"
    }
    
    return {
        "metadata": metadata,
        "data": data
    }

def main():
    """
    Función principal mejorada para ML pipeline.
    """
    try:
        logger.info("Starting crypto data extraction for ML pipeline")
        
        # Extraer datos
        raw_data = fetch_coingecko_top(n=250)
        
        # Añadir metadatos para ML
        data_with_metadata = add_extraction_metadata(raw_data)
        
        # Generar nombre de archivo con timestamp
        now_utc = datetime.now(timezone.utc)
        fname = OUT_DIR / f"coins_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Guardar datos con metadatos
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Successfully saved {len(raw_data)} coins to {fname}")
        logger.info(f"📊 Data includes: price_change_1h, price_change_24h, price_change_7d")
        
        return fname
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
