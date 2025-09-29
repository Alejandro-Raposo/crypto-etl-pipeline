from google.cloud import bigquery
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

# ------------------------------
# Config
# ------------------------------
PROC_DIR = Path("./data/processed")
PROJECT = "crypto-etl-proyect"      
DATASET = "crypto_dataset"
TABLE = "prices"

# Load .env from parent folder
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

CREDENTIALS_PATH = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not CREDENTIALS_PATH or not Path(CREDENTIALS_PATH).exists():
    raise FileNotFoundError(
        f"Service account JSON not found. Check ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE in {dotenv_path}"
    )

# ------------------------------
# Helper functions
# ------------------------------
def latest_parquet():
    files = sorted(PROC_DIR.glob("coins_processed_*.parquet"))
    if not files:
        raise FileNotFoundError("No processed files.")
    return files[-1]

def ensure_dataset(client):
    ds_ref = client.dataset(DATASET)
    try:
        client.get_dataset(ds_ref)
        print("Dataset exists:", DATASET)
    except Exception:
        ds = bigquery.Dataset(ds_ref)
        ds.location = "EU" 
        client.create_dataset(ds)
        print("Created dataset:", DATASET)

# ------------------------------
# Load to BigQuery
# ------------------------------
def load_to_bq():
    client = bigquery.Client.from_service_account_json(CREDENTIALS_PATH, project=PROJECT)
    ensure_dataset(client)
    
    file = latest_parquet()
    df = pd.read_parquet(file)
    
    table_ref = client.dataset(DATASET).table(TABLE)
    
    # Configuración de carga más flexible para ML
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,  # Detectar schema automáticamente
        source_format=bigquery.SourceFormat.PARQUET
    )
    
    # Intentar cargar, y si falla, crear tabla nueva con sufijo ML
    try:
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  
        print(f"Loaded {len(df)} rows to {PROJECT}.{DATASET}.{TABLE}")
    except Exception as e:
        print(f"Error loading to existing table: {e}")
        print("Creating new ML-ready table...")
        
        # Crear tabla nueva con sufijo _ml
        ml_table = f"{TABLE}_ml"
        ml_table_ref = client.dataset(DATASET).table(ml_table)
        
        job_config.write_disposition = "WRITE_TRUNCATE"  # Sobrescribir si existe
        job = client.load_table_from_dataframe(df, ml_table_ref, job_config=job_config)
        job.result()
        
        print(f"Loaded {len(df)} rows to {PROJECT}.{DATASET}.{ml_table}")
        print(f"New ML table created: {ml_table}")

# ------------------------------
if __name__ == "__main__":
    load_to_bq()
