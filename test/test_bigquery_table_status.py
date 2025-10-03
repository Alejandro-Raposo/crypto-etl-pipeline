import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(f"Credenciales no encontradas: {credentials_path}")

PROJECT = "crypto-etl-proyect"
DATASET = "crypto_dataset"
TABLE_ML = "prices_ml"
TABLE_REGULAR = "prices"

client = bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)

def check_table_exists(table_name):
    table_ref = f"{PROJECT}.{DATASET}.{table_name}"
    try:
        table = client.get_table(table_ref)
        return True, table
    except Exception as e:
        return False, str(e)

def get_table_info(table_name):
    exists, result = check_table_exists(table_name)
    if not exists:
        print(f"[X] Tabla {table_name} NO existe")
        print(f"   Error: {result}")
        return None
    
    table = result
    print(f"[OK] Tabla {table_name} existe")
    print(f"   - Filas: {table.num_rows:,}")
    print(f"   - Tamaño: {table.num_bytes / 1024 / 1024:.2f} MB")
    print(f"   - Columnas: {len(table.schema)}")
    print(f"   - Creada: {table.created}")
    print(f"   - Ultima modificacion: {table.modified}")
    
    query = f"""
    SELECT COUNT(*) as total_rows,
           COUNT(DISTINCT id) as unique_coins,
           MIN(last_updated) as oldest_record,
           MAX(last_updated) as newest_record
    FROM `{PROJECT}.{DATASET}.{table_name}`
    """
    
    try:
        df = client.query(query).to_dataframe()
        print(f"   - Registros totales: {df['total_rows'].iloc[0]:,}")
        print(f"   - Criptomonedas unicas: {df['unique_coins'].iloc[0]:,}")
        print(f"   - Registro mas antiguo: {df['oldest_record'].iloc[0]}")
        print(f"   - Registro mas reciente: {df['newest_record'].iloc[0]}")
    except Exception as e:
        print(f"   [!] Error al consultar datos: {e}")
    
    print("\n   Schema (primeras 10 columnas):")
    for i, field in enumerate(table.schema[:10]):
        print(f"     {i+1}. {field.name} ({field.field_type})")
    if len(table.schema) > 10:
        print(f"     ... y {len(table.schema) - 10} columnas más")
    
    return table

def check_local_processed_files():
    proc_dir = Path(__file__).parent.parent / "data/processed"
    files = sorted(proc_dir.glob("coins_processed_*.parquet"))
    
    print(f"\n[FILES] Archivos procesados localmente: {len(files)}")
    if files:
        latest = files[-1]
        df = pd.read_parquet(latest)
        print(f"   - Ultimo archivo: {latest.name}")
        print(f"   - Filas: {len(df):,}")
        print(f"   - Columnas: {len(df.columns)}")
        print(f"   - Columnas ML: {df.columns.tolist()[:15]}...")
        return df
    return None

print("="*60)
print("VERIFICACION DE TABLAS EN BIGQUERY")
print("="*60)

print(f"\nProyecto: {PROJECT}")
print(f"Dataset: {DATASET}\n")

print("-" * 60)
print("TABLA: prices_ml")
print("-" * 60)
table_ml = get_table_info(TABLE_ML)

print("\n" + "-" * 60)
print("TABLA: prices (comparacion)")
print("-" * 60)
table_regular = get_table_info(TABLE_REGULAR)

print("\n" + "=" * 60)
print("COMPARACION CON DATOS LOCALES")
print("=" * 60)
local_df = check_local_processed_files()

print("\n" + "=" * 60)
print("DIAGNOSTICO")
print("=" * 60)

if table_ml and table_ml.num_rows > 0:
    print("[OK] La tabla prices_ml existe y contiene datos")
    if local_df is not None:
        if table_ml.num_rows >= len(local_df):
            print("[OK] La tabla tiene al menos los mismos registros que el ultimo archivo local")
        else:
            print(f"[!] La tabla tiene menos registros ({table_ml.num_rows}) que el archivo local ({len(local_df)})")
elif table_ml and table_ml.num_rows == 0:
    print("[!] La tabla prices_ml existe pero esta VACIA")
    print("   Posibles causas:")
    print("   1. El pipeline nunca se ejecuto completamente")
    print("   2. Hubo un error durante la carga")
    print("   3. Los datos se cargaron a otra tabla")
else:
    print("[X] La tabla prices_ml NO EXISTE")
    print("   Posibles causas:")
    print("   1. El pipeline fallo al crear la tabla")
    print("   2. La tabla nunca se creo porque no hubo errores en 'prices'")
    print("   3. El script load.py usa 'prices' por defecto y solo crea 'prices_ml' si hay error")

print("\nRECOMENDACION:")
if not table_ml or (table_ml and table_ml.num_rows == 0):
    print("   Ejecutar: python scripts/run_ml_pipeline.py")
    print("   O ejecutar manualmente: python scripts/load.py")

