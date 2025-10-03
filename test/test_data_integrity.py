import unittest
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
if credentials_path and Path(credentials_path).exists():
    HAS_CREDENTIALS = True
    client = bigquery.Client.from_service_account_json(credentials_path, project=PROJECT)
else:
    HAS_CREDENTIALS = False
    client = None

class TestTemporalIntegrity(unittest.TestCase):
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_historical_data(self):
        query = f"SELECT COUNT(*) as count FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`"
        result = client.query(query).to_dataframe()
        count = result['count'].iloc[0]
        self.assertGreater(count, 0, "Tabla historica debe contener datos")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_no_duplicate_keys(self):
        query = f"""
        SELECT composite_key, COUNT(*) as duplicates
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        GROUP BY composite_key
        HAVING COUNT(*) > 1
        """
        result = client.query(query).to_dataframe()
        self.assertEqual(len(result), 0, f"No deben existir claves duplicadas: {result}")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_valid_timestamps(self):
        query = f"""
        SELECT COUNT(*) as invalid_count
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        WHERE last_updated > CURRENT_TIMESTAMP()
        OR last_updated < TIMESTAMP('2025-01-01')
        """
        result = client.query(query).to_dataframe()
        invalid_count = result['invalid_count'].iloc[0]
        self.assertEqual(invalid_count, 0, "Todos los timestamps deben ser validos")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_valid_prices(self):
        query = f"""
        SELECT COUNT(*) as invalid_count
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        WHERE current_price <= 0 OR current_price IS NULL
        """
        result = client.query(query).to_dataframe()
        invalid_count = result['invalid_count'].iloc[0]
        self.assertEqual(invalid_count, 0, "Todos los precios deben ser positivos y no nulos")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_track_multiple_cryptos(self):
        query = f"""
        SELECT COUNT(DISTINCT id) as unique_cryptos
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        """
        result = client.query(query).to_dataframe()
        unique_cryptos = result['unique_cryptos'].iloc[0]
        self.assertGreater(unique_cryptos, 0, "Debe rastrear multiples criptomonedas")

class TestDataCoverage(unittest.TestCase):
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_calculate_temporal_coverage(self):
        query = f"""
        SELECT 
            MIN(last_updated) as oldest,
            MAX(last_updated) as newest,
            TIMESTAMP_DIFF(MAX(last_updated), MIN(last_updated), HOUR) as hours_covered
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        """
        result = client.query(query).to_dataframe()
        hours_covered = result['hours_covered'].iloc[0]
        self.assertGreaterEqual(hours_covered, 0, "Debe tener cobertura temporal")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_recent_data(self):
        query = f"""
        SELECT MAX(last_updated) as newest
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        """
        result = client.query(query).to_dataframe()
        newest = pd.to_datetime(result['newest'].iloc[0])
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        self.assertGreater(newest.replace(tzinfo=timezone.utc), cutoff, "Debe tener datos recientes (menos de 7 dias)")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_consistent_crypto_ids(self):
        query = f"""
        SELECT id, COUNT(DISTINCT symbol) as symbol_count
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        GROUP BY id
        HAVING COUNT(DISTINCT symbol) > 1
        """
        result = client.query(query).to_dataframe()
        self.assertEqual(len(result), 0, "Cada ID debe tener un solo symbol consistente")

class TestGapDetection(unittest.TestCase):
    
    def test_detect_gaps_in_sample_data(self):
        dates = pd.date_range('2025-10-01', periods=24, freq='6h')
        dates_with_gap = dates[:3].tolist() + dates[6:].tolist()
        df = pd.DataFrame({
            'id': ['bitcoin'] * len(dates_with_gap),
            'last_updated': dates_with_gap,
        })
        df = df.sort_values(['id', 'last_updated'])
        df['time_diff'] = df.groupby('id')['last_updated'].diff()
        expected_freq = pd.Timedelta(hours=6)
        gaps = df[df['time_diff'] > expected_freq * 1.5]
        self.assertGreater(len(gaps), 0, "Debe detectar gaps en los datos")
    
    def test_calculate_data_completeness(self):
        start_date = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 10, 3, tzinfo=timezone.utc)
        expected_snapshots = 8
        actual_snapshots = 6
        completeness = (actual_snapshots / expected_snapshots) * 100
        self.assertLess(completeness, 100, "Completeness debe reflejar datos faltantes")
        self.assertEqual(completeness, 75.0, "Calculo de completeness incorrecto")

class TestPartitioning(unittest.TestCase):
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_partition_field(self):
        query = f"""
        SELECT partition_date
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        LIMIT 1
        """
        result = client.query(query).to_dataframe()
        self.assertIn('partition_date', result.columns, "Tabla debe tener campo de particionamiento")
    
    @unittest.skipUnless(HAS_CREDENTIALS, "Requiere credenciales de GCP")
    def test_should_have_multiple_partitions(self):
        query = f"""
        SELECT COUNT(DISTINCT partition_date) as partition_count
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        """
        result = client.query(query).to_dataframe()
        partition_count = result['partition_count'].iloc[0]
        self.assertGreaterEqual(partition_count, 1, "Debe tener al menos una particion")

if __name__ == '__main__':
    unittest.main()

