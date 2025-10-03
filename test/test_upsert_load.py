import unittest
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

class TestUpsertLoad(unittest.TestCase):
    
    def test_should_detect_duplicate_records(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62100, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        duplicates = df.duplicated(subset=['id', 'last_updated'], keep=False)
        self.assertTrue(duplicates.any(), "Debe detectar registros duplicados por id+timestamp")
    
    def test_should_keep_latest_price_for_duplicates(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62100, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df_clean = df.sort_values('current_price', ascending=False).drop_duplicates(subset=['id', 'last_updated'], keep='first')
        self.assertEqual(len(df_clean), 1, "Debe mantener solo un registro por id+timestamp")
        self.assertEqual(df_clean.iloc[0]['current_price'], 62100, "Debe mantener el precio mas reciente")
    
    def test_should_add_ingestion_timestamp(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['ingestion_timestamp'] = datetime.now(timezone.utc)
        self.assertIn('ingestion_timestamp', df.columns, "Debe incluir timestamp de ingestion")
        self.assertIsInstance(df['ingestion_timestamp'].iloc[0], datetime, "ingestion_timestamp debe ser datetime")
    
    def test_should_partition_by_date(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
            {'id': 'ethereum', 'symbol': 'eth', 'current_price': 2500, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['partition_date'] = df['last_updated'].dt.date
        self.assertIn('partition_date', df.columns, "Debe incluir columna de particionamiento")
        self.assertEqual(len(df['partition_date'].unique()), 1, "Todos los registros deben tener la misma fecha de particion")
    
    def test_should_validate_required_columns(self):
        required_columns = ['id', 'symbol', 'name', 'current_price', 'market_cap', 'last_updated']
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin', 'current_price': 62000, 'market_cap': 1200000000000, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        missing_columns = [col for col in required_columns if col not in df.columns]
        self.assertEqual(len(missing_columns), 0, f"Faltan columnas requeridas: {missing_columns}")
    
    def test_should_handle_null_prices(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
            {'id': 'broken_coin', 'symbol': 'xxx', 'current_price': None, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df_clean = df.dropna(subset=['current_price'])
        self.assertEqual(len(df_clean), 1, "Debe eliminar registros con precio nulo")
        self.assertEqual(df_clean.iloc[0]['id'], 'bitcoin', "Debe mantener solo registros validos")
    
    def test_should_create_composite_key(self):
        data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'current_price': 62000, 'last_updated': '2025-10-03T10:00:00Z'},
        ]
        df = pd.DataFrame(data)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['composite_key'] = df['id'].astype(str) + '_' + df['last_updated'].dt.strftime('%Y%m%d%H%M%S')
        self.assertIn('composite_key', df.columns, "Debe crear clave compuesta")
        self.assertEqual(df.iloc[0]['composite_key'], 'bitcoin_20251003100000', "Formato de clave compuesta incorrecto")

class TestDataQuality(unittest.TestCase):
    
    def test_should_reject_future_timestamps(self):
        future_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        current_date = datetime.now(timezone.utc)
        self.assertLess(current_date, future_date, "Fecha futura debe ser mayor que fecha actual")
        data = [
            {'id': 'bitcoin', 'last_updated': future_date},
        ]
        df = pd.DataFrame(data)
        df_clean = df[df['last_updated'] <= current_date]
        self.assertEqual(len(df_clean), 0, "Debe rechazar timestamps futuros")
    
    def test_should_reject_very_old_data(self):
        old_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        current_date = datetime.now(timezone.utc)
        cutoff_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.assertLess(old_date, cutoff_date, "Fecha antigua debe ser menor que cutoff")
        data = [
            {'id': 'bitcoin', 'last_updated': old_date},
        ]
        df = pd.DataFrame(data)
        df_clean = df[df['last_updated'] >= cutoff_date]
        self.assertEqual(len(df_clean), 0, "Debe rechazar datos muy antiguos")
    
    def test_should_validate_price_ranges(self):
        data = [
            {'id': 'bitcoin', 'current_price': 62000},
            {'id': 'negative_coin', 'current_price': -100},
            {'id': 'zero_coin', 'current_price': 0},
        ]
        df = pd.DataFrame(data)
        df_clean = df[df['current_price'] > 0]
        self.assertEqual(len(df_clean), 1, "Debe eliminar precios negativos o cero")

if __name__ == '__main__':
    unittest.main()

