import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

PROJECT = "crypto-etl-proyect"
DATASET = "crypto_dataset"
TABLE_HISTORICAL = "prices_historical"

credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(f"Credenciales no encontradas: {credentials_path}")

client = bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def get_general_stats():
    query = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as unique_cryptos,
        COUNT(DISTINCT partition_date) as unique_dates,
        MIN(last_updated) as oldest_record,
        MAX(last_updated) as newest_record,
        TIMESTAMP_DIFF(MAX(last_updated), MIN(last_updated), HOUR) as hours_covered
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    """
    df = client.query(query).to_dataframe()
    return df.iloc[0]

def get_top_cryptos_by_coverage():
    query = f"""
    SELECT 
        id,
        symbol,
        COUNT(*) as snapshots,
        MIN(last_updated) as first_record,
        MAX(last_updated) as last_record
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    GROUP BY id, symbol
    ORDER BY snapshots DESC
    LIMIT 10
    """
    return client.query(query).to_dataframe()

def get_snapshots_per_date():
    query = f"""
    SELECT 
        partition_date,
        COUNT(*) as snapshots,
        COUNT(DISTINCT id) as unique_cryptos
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    GROUP BY partition_date
    ORDER BY partition_date DESC
    LIMIT 30
    """
    return client.query(query).to_dataframe()

def detect_gaps(hours_threshold=7):
    query = f"""
    WITH ordered_data AS (
        SELECT 
            id,
            last_updated,
            LAG(last_updated) OVER (PARTITION BY id ORDER BY last_updated) as prev_timestamp
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    )
    SELECT 
        id,
        prev_timestamp,
        last_updated,
        TIMESTAMP_DIFF(last_updated, prev_timestamp, HOUR) as hours_gap
    FROM ordered_data
    WHERE prev_timestamp IS NOT NULL 
        AND TIMESTAMP_DIFF(last_updated, prev_timestamp, HOUR) > {hours_threshold}
    ORDER BY hours_gap DESC
    LIMIT 20
    """
    return client.query(query).to_dataframe()

def get_data_quality_metrics():
    query = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNTIF(current_price IS NULL) as null_prices,
        COUNTIF(current_price <= 0) as invalid_prices,
        COUNTIF(market_cap IS NULL) as null_market_caps,
        COUNTIF(total_volume IS NULL) as null_volumes
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    """
    df = client.query(query).to_dataframe()
    return df.iloc[0]

def calculate_completeness_score(stats):
    current_time = datetime.now(timezone.utc)
    oldest = pd.to_datetime(stats['oldest_record']).replace(tzinfo=timezone.utc)
    hours_elapsed = (current_time - oldest).total_seconds() / 3600
    expected_snapshots_6h = max(1, hours_elapsed / 6)
    actual_snapshots = stats['total_records'] / max(1, stats['unique_cryptos'])
    completeness = min(100, (actual_snapshots / expected_snapshots_6h) * 100)
    return completeness

def display_dashboard():
    print_header("CRYPTO ETL PIPELINE - MONITORING DASHBOARD")
    
    print("\n[1] ESTADISTICAS GENERALES")
    stats = get_general_stats()
    print(f"  Total registros:      {stats['total_records']:,}")
    print(f"  Cryptos unicas:       {stats['unique_cryptos']:,}")
    print(f"  Fechas unicas:        {stats['unique_dates']}")
    print(f"  Horas cubiertas:      {stats['hours_covered']:.1f} horas")
    print(f"  Registro mas antiguo: {stats['oldest_record']}")
    print(f"  Registro mas reciente: {stats['newest_record']}")
    
    completeness = calculate_completeness_score(stats)
    print(f"\n  Completeness Score:   {completeness:.1f}%")
    if completeness >= 90:
        print("  Estado: EXCELENTE")
    elif completeness >= 70:
        print("  Estado: BUENO")
    elif completeness >= 50:
        print("  Estado: ACEPTABLE")
    else:
        print("  Estado: NECESITA MEJORAS")
    
    print("\n[2] CALIDAD DE DATOS")
    quality = get_data_quality_metrics()
    print(f"  Precios nulos:        {quality['null_prices']:,} ({quality['null_prices']/quality['total_records']*100:.2f}%)")
    print(f"  Precios invalidos:    {quality['invalid_prices']:,} ({quality['invalid_prices']/quality['total_records']*100:.2f}%)")
    print(f"  Market caps nulos:    {quality['null_market_caps']:,} ({quality['null_market_caps']/quality['total_records']*100:.2f}%)")
    print(f"  Volumenes nulos:      {quality['null_volumes']:,} ({quality['null_volumes']/quality['total_records']*100:.2f}%)")
    
    print("\n[3] TOP 10 CRYPTOS POR COBERTURA")
    top_cryptos = get_top_cryptos_by_coverage()
    for idx, row in top_cryptos.iterrows():
        print(f"  {idx+1}. {row['symbol'].upper():6} ({row['id']:15}) - {row['snapshots']:3} snapshots")
    
    print("\n[4] SNAPSHOTS POR FECHA (ultimos 30 dias)")
    snapshots_per_date = get_snapshots_per_date()
    for idx, row in snapshots_per_date.iterrows():
        print(f"  {row['partition_date']} - {row['snapshots']:4} snapshots ({row['unique_cryptos']:3} cryptos)")
    
    print("\n[5] DETECCION DE GAPS (>7 horas)")
    gaps = detect_gaps(hours_threshold=7)
    if len(gaps) == 0:
        print("  No se detectaron gaps significativos")
    else:
        print(f"  Se detectaron {len(gaps)} gaps:")
        for idx, row in gaps.head(10).iterrows():
            print(f"  - {row['id']:15} - Gap de {row['hours_gap']:.1f} horas")
    
    print_header("FIN DEL DASHBOARD")
    
    print("\nRECOMENDACIONES:")
    if completeness < 90:
        print("  - Considera aumentar frecuencia de ejecucion del pipeline")
    if len(gaps) > 0:
        print("  - Revisar logs del pipeline en las fechas con gaps detectados")
    if quality['null_prices'] > 0 or quality['invalid_prices'] > 0:
        print("  - Revisar validacion de datos en extract/transform")
    if stats['hours_covered'] < 168:
        print("  - Acumula mas datos historicos (minimo 7 dias recomendado para ML)")

if __name__ == "__main__":
    display_dashboard()

