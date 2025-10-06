import unittest
import pandas as pd
import numpy as np
from datetime import datetime

class TestMLDataLoader(unittest.TestCase):
    
    def test_debe_cargar_datos_de_bigquery(self):
        """Test que valida carga básica de datos"""
        from ml.data_loader import load_crypto_data
        df = load_crypto_data(crypto_id='bitcoin', days=7)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    def test_debe_tener_columnas_requeridas(self):
        """Test que valida schema de datos"""
        from ml.data_loader import load_crypto_data
        required_cols = [
            'id', 'last_updated', 'current_price',
            'price_lag_1h', 'price_ma_6h', 'rsi_14h'
        ]
        df = load_crypto_data(crypto_id='bitcoin', days=7)
        for col in required_cols:
            self.assertIn(col, df.columns, f"Falta columna requerida: {col}")
    
    def test_debe_ordenar_por_fecha(self):
        """Test que valida ordenamiento temporal"""
        from ml.data_loader import load_crypto_data
        df = load_crypto_data(crypto_id='bitcoin', days=7)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        is_sorted = (df['last_updated'].diff()[1:] >= pd.Timedelta(0)).all()
        self.assertTrue(is_sorted, "Datos deben estar ordenados cronológicamente")
    
    def test_debe_eliminar_nulls_en_features(self):
        """Test que valida limpieza de datos"""
        from ml.data_loader import load_crypto_data
        df = load_crypto_data(crypto_id='bitcoin', days=7)
        key_features = ['price_lag_1h', 'price_ma_6h', 'rsi_14h']
        for col in key_features:
            if col in df.columns:
                null_count = df[col].isna().sum()
                self.assertEqual(null_count, 0, f"No debe haber nulls en {col}")
    
    def test_debe_validar_crypto_id(self):
        """Test que valida entrada de crypto_id"""
        from ml.data_loader import load_crypto_data
        with self.assertRaises(ValueError):
            load_crypto_data(crypto_id='', days=7)
        with self.assertRaises(ValueError):
            load_crypto_data(crypto_id=None, days=7)

if __name__ == '__main__':
    unittest.main()

