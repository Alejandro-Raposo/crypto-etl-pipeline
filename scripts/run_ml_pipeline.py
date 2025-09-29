#!/usr/bin/env python3
"""
Pipeline automatizado para ML de criptomonedas.
Ejecuta todo el flujo ETL optimizado para machine learning.
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

# Añadir el directorio de scripts al path
sys.path.append(str(Path(__file__).parent))

# Importar módulos locales
try:
    from extract import main as extract_main
    from transform import main as transform_main
    from load import load_to_bq
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """
    Ejecuta el pipeline completo ETL para ML.
    """
    start_time = time.time()
    
    try:
        logger.info("Starting ML Pipeline for Crypto Data")
        logger.info("="*50)
        
        # STEP 1: Extract
        logger.info("STEP 1/3: Extracting data from CoinGecko API...")
        raw_file = extract_main()
        logger.info(f"Extraction completed: {raw_file}")
        
        # STEP 2: Transform 
        logger.info("STEP 2/3: Transforming data with ML features...")
        processed_file = transform_main()
        logger.info(f"Transformation completed: {processed_file}")
        
        # STEP 3: Load
        logger.info("STEP 3/3: Loading to BigQuery...")
        load_to_bq()
        logger.info("Load completed to BigQuery")
        
        # Pipeline summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("="*50)
        logger.info("ML PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {duration:.2f} seconds")
        logger.info(f"Data ready for ML training at: {processed_file}")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check logs for detailed error information")
        return False

def pipeline_health_check():
    """
    Verifica que el pipeline esté funcionando correctamente.
    """
    logger.info("Running pipeline health check...")
    
    # Verificar directorios
    data_dirs = ['data/raw', 'data/processed']
    for dir_path in data_dirs:
        if not Path(dir_path).exists():
            logger.error(f"Directory missing: {dir_path}")
            return False
        logger.info(f"Directory exists: {dir_path}")
    
    # Verificar archivos recientes
    raw_files = list(Path('data/raw').glob('coins_*.json'))
    processed_files = list(Path('data/processed').glob('coins_processed_*.parquet'))
    
    logger.info(f"Raw files: {len(raw_files)}")
    logger.info(f"Processed files: {len(processed_files)}")
    
    if raw_files and processed_files:
        latest_raw = max(raw_files, key=lambda x: x.stat().st_mtime)
        latest_processed = max(processed_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Latest raw file: {latest_raw.name}")
        logger.info(f"Latest processed file: {latest_processed.name}")
    
    logger.info("Health check completed")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Pipeline for Crypto Data')
    parser.add_argument('--health-check', action='store_true', 
                       help='Run pipeline health check')
    parser.add_argument('--extract-only', action='store_true',
                       help='Run only extraction step')
    parser.add_argument('--transform-only', action='store_true',
                       help='Run only transformation step')
    
    args = parser.parse_args()
    
    if args.health_check:
        pipeline_health_check()
    elif args.extract_only:
        extract_main()
    elif args.transform_only:
        transform_main()
    else:
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
