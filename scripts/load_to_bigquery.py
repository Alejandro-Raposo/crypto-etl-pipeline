import os
import json
import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
# .env is one folder back from this script
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Get service account path from .env
credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")
if not credentials_path or not Path(credentials_path).exists():
    raise FileNotFoundError(
        f"Service account JSON not found. CheckETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE in {dotenv_path}"
    )
print("Using credentials:", credentials_path)

# ------------------------------
# Initialize BigQuery client
# ------------------------------
client = bigquery.Client.from_service_account_json(credentials_path)

# ------------------------------
# Load latest JSON from raw data folder
# ------------------------------
RAW_DIR = Path(__file__).parent.parent / "data/raw"
latest_file = max(RAW_DIR.glob("coins_*.json"), key=os.path.getctime)
print(f"📂 Loading data from: {latest_file}")

with open(latest_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"✅ DataFrame loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Optional: Add ingestion timestamp
df["ingestion_time"] = pd.Timestamp.utcnow()

# ------------------------------
# BigQuery destination
# ------------------------------
dataset_id = "crypto_dataset"      # 👈 replace with your dataset name
table_id = "crypto_markets"      # 👈 table name
table_ref = f"{client.project}.{dataset_id}.{table_id}"

# Make sure dataset exists
dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
client.create_dataset(dataset_ref, exists_ok=True)

# Upload DataFrame to BigQuery
job = client.load_table_from_dataframe(
    df,
    table_ref,
    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"  # append new data, change to WRITE_TRUNCATE to overwrite
    )
)
job.result()  # wait for the job to finish

print(f"✅ Uploaded {job.output_rows} rows to {table_ref}")