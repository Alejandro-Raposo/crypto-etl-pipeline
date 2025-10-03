import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from google.cloud import bigquery
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

PROC_DIR = Path(__file__).parent.parent / "data/processed"
PROJECT = "crypto-etl-proyect"
DATASET = "crypto_dataset"
TABLE_HISTORICAL = "prices_historical"

credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(f"Credenciales no encontradas: {credentials_path}")

def get_latest_parquet():
    files = sorted(PROC_DIR.glob("coins_processed_*.parquet"))
    if not files:
        raise FileNotFoundError("No hay archivos procesados")
    return files[-1]

def prepare_data_for_upsert(df):
    df = df.copy()
    df = df.dropna(subset=['current_price'])
    df = df[df['current_price'] > 0]
    if 'last_updated' in df.columns:
        df['last_updated'] = pd.to_datetime(df['last_updated'], utc=True)
        current_time = datetime.now(timezone.utc)
        cutoff_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df = df[df['last_updated'] <= current_time]
        df = df[df['last_updated'] >= cutoff_date]
    df['ingestion_timestamp'] = pd.Timestamp.now(tz='UTC')
    df['partition_date'] = df['last_updated'].dt.date
    df['composite_key'] = df['id'].astype(str) + '_' + df['last_updated'].dt.strftime('%Y%m%d%H%M%S')
    df = df.sort_values('current_price', ascending=False).drop_duplicates(subset=['id', 'last_updated'], keep='first')
    return df

def create_historical_table(client):
    dataset_ref = client.dataset(DATASET)
    table_ref = dataset_ref.table(TABLE_HISTORICAL)
    try:
        client.get_table(table_ref)
        logger.info(f"Tabla {TABLE_HISTORICAL} ya existe")
        return
    except Exception:
        pass
    schema = [
        bigquery.SchemaField("composite_key", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("current_price", "FLOAT"),
        bigquery.SchemaField("market_cap", "INTEGER"),
        bigquery.SchemaField("total_volume", "FLOAT"),
        bigquery.SchemaField("market_cap_rank", "INTEGER"),
        bigquery.SchemaField("price_change_24h", "FLOAT"),
        bigquery.SchemaField("last_updated", "TIMESTAMP"),
        bigquery.SchemaField("price_change_1h", "FLOAT"),
        bigquery.SchemaField("price_change_7d", "FLOAT"),
        bigquery.SchemaField("high_24h", "FLOAT"),
        bigquery.SchemaField("low_24h", "FLOAT"),
        bigquery.SchemaField("partition_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("hour", "INTEGER"),
        bigquery.SchemaField("market_cap_millions", "FLOAT"),
        bigquery.SchemaField("price_log", "FLOAT"),
        bigquery.SchemaField("market_cap_log", "FLOAT"),
        bigquery.SchemaField("volume_log", "FLOAT"),
        bigquery.SchemaField("volume_to_market_cap_ratio", "FLOAT"),
        bigquery.SchemaField("volume_millions", "FLOAT"),
        bigquery.SchemaField("market_cap_category", "STRING"),
        bigquery.SchemaField("price_volatility_24h", "FLOAT"),
        bigquery.SchemaField("market_cap_rank_normalized", "FLOAT"),
        bigquery.SchemaField("extraction_timestamp", "STRING"),
        bigquery.SchemaField("pipeline_version", "STRING"),
    ]
    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="partition_date",
    )
    table.clustering_fields = ["id", "partition_date"]
    client.create_table(table)
    logger.info(f"Tabla {TABLE_HISTORICAL} creada con particionamiento por fecha")

def upsert_data(client, df):
    table_ref = client.dataset(DATASET).table(TABLE_HISTORICAL)
    try:
        existing_data_query = f"""
        SELECT * FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        """
        existing_df = client.query(existing_data_query).to_dataframe()
        logger.info(f"Registros existentes: {len(existing_df)}")
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.sort_values('ingestion_timestamp', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['composite_key'], keep='first')
        logger.info(f"Registros despues de deduplicacion: {len(combined_df)}")
    except Exception as e:
        logger.info(f"No hay datos previos o error al leer: {e}")
        combined_df = df
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(combined_df, table_ref, job_config=job_config)
    job.result()
    logger.info(f"UPSERT completado: {len(combined_df)} registros totales")

def load_historical():
    client = bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)
    logger.info("Iniciando carga historica")
    create_historical_table(client)
    file = get_latest_parquet()
    logger.info(f"Procesando archivo: {file.name}")
    df = pd.read_parquet(file)
    logger.info(f"Registros leidos: {len(df)}")
    df_clean = prepare_data_for_upsert(df)
    logger.info(f"Registros limpios: {len(df_clean)}")
    upsert_data(client, df_clean)
    stats_query = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as unique_cryptos,
        MIN(last_updated) as oldest_record,
        MAX(last_updated) as newest_record,
        COUNT(DISTINCT partition_date) as unique_dates
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    """
    stats = client.query(stats_query).to_dataframe()
    logger.info(f"Estadisticas tabla historica:")
    logger.info(f"  Total registros: {stats['total_records'].iloc[0]:,}")
    logger.info(f"  Cryptos unicas: {stats['unique_cryptos'].iloc[0]:,}")
    logger.info(f"  Fechas unicas: {stats['unique_dates'].iloc[0]:,}")
    logger.info(f"  Rango: {stats['oldest_record'].iloc[0]} - {stats['newest_record'].iloc[0]}")

if __name__ == "__main__":
    load_historical()

