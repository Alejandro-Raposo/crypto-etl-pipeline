"""
Google Cloud Function para ejecutar el pipeline de ML de criptomonedas.
Se ejecuta automáticamente con Cloud Scheduler.
"""

import functions_framework
import os
import sys
import tempfile
import json
from pathlib import Path

# Importar módulos del pipeline
sys.path.append(str(Path(__file__).parent))

def setup_temp_directories():
    """Crear directorios temporales para datos."""
    temp_dir = Path(tempfile.gettempdir()) / "crypto-pipeline"
    raw_dir = temp_dir / "data" / "raw"
    processed_dir = temp_dir / "data" / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir

@functions_framework.http
def crypto_ml_pipeline(request):
    """
    Cloud Function HTTP endpoint para ejecutar el pipeline.
    """
    try:
        # Configurar entorno temporal
        temp_dir = setup_temp_directories()
        os.chdir(temp_dir)
        
        # Importar y ejecutar pipeline
        from scripts.extract import fetch_coingecko_top, add_extraction_metadata
        from scripts.transform import transform
        from scripts.load import load_to_bq
        
        # STEP 1: Extract
        print("🚀 Starting extraction...")
        raw_data = fetch_coingecko_top(n=250)
        data_with_metadata = add_extraction_metadata(raw_data)
        
        # STEP 2: Transform  
        print("🔄 Starting transformation...")
        df = transform(data_with_metadata["data"], data_with_metadata["metadata"])
        
        # STEP 3: Load directly to BigQuery (sin archivos temporales)
        print("📤 Loading to BigQuery...")
        
        # Configurar credenciales desde variable de entorno
        from google.cloud import bigquery
        import pandas as pd
        
        client = bigquery.Client()
        table_ref = client.dataset("crypto_dataset").table("prices_ml")
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=True
        )
        
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        
        result = {
            "status": "success",
            "message": f"Pipeline completed successfully. Loaded {len(df)} rows.",
            "rows_processed": len(df),
            "features_count": len(df.columns),
            "timestamp": data_with_metadata["metadata"]["extraction_timestamp"]
        }
        
        print(f"✅ Success: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error", 
            "message": f"Pipeline failed: {str(e)}",
            "error_type": type(e).__name__
        }
        print(f"❌ Error: {error_result}")
        return error_result, 500

@functions_framework.cloud_event
def crypto_ml_pipeline_scheduled(cloud_event):
    """
    Cloud Function que se ejecuta con Cloud Scheduler.
    """
    print(f"🕐 Scheduled execution triggered: {cloud_event}")
    
    # Simular request HTTP para reutilizar la lógica
    class MockRequest:
        pass
    
    return crypto_ml_pipeline(MockRequest())
