import os
from dotenv import load_dotenv
from google.cloud import bigquery

# Load .env from parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Get the path to your service account JSON
credentials_path = os.getenv("ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE")

# Optional: check if it loaded correctly
print("Credentials path:", credentials_path)

# Initialize BigQuery client
client = bigquery.Client.from_service_account_json(credentials_path)

# Test connection by listing datasets
project_id = client.project
datasets = list(client.list_datasets(project_id))

if datasets:
    print(f"Datasets in project {project_id}:")
    for dataset in datasets:
        print(f" - {dataset.dataset_id}")
else:
    print(f"No datasets found in project {project_id}.")
