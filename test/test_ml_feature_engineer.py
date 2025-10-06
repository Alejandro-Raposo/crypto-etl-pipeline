import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestMLFeatureEngineer(unittest.TestCase):
    
    def setUp(self):
        """Crear datos de prueba"""
        dates = pd.date_range('2025-10-01', periods=24, freq='h')
        self.df = pd.DataFrame({
            'id': ['bitcoin'] * 24,
            'last_updated': dates,
            'current_price': [62000 + i * 100 for i in range(24)],
            'price_lag_1h': [62000 + i * 100 for i in range(-1, 23)],
            'price_ma_6h': [62000 + i * 50 for i in range(24)],
            'rsi_14h': [50 + i for i in range(24)],
            'volatility_24h': [2.5] * 24,
        })
    
    def test_debe_crear_target_binario(self):
        """Test que valida creación de variable target"""
        from ml.feature_engineer import create_target
        df_with_target = create_target(self.df.copy())
        self.assertIn('target', df_with_target.columns)
        self.assertTrue(df_with_target['target'].isin([0, 1]).all())
    
    def test_target_debe_ser_1_cuando_precio_sube(self):
        """Test que valida lógica de target=1 (sube)"""
        from ml.feature_engineer import create_target
        df = pd.DataFrame({
            'current_price': [100, 105],
            'last_updated': pd.date_range('2025-10-01', periods=2, freq='h')
        })
        df_with_target = create_target(df)
        self.assertEqual(df_with_target.iloc[0]['target'], 1)
    
    def test_target_debe_ser_0_cuando_precio_baja(self):
        """Test que valida lógica de target=0 (baja)"""
        from ml.feature_engineer import create_target
        df = pd.DataFrame({
            'current_price': [100, 95],
            'last_updated': pd.date_range('2025-10-01', periods=2, freq='h')
        })
        df_with_target = create_target(df)
        self.assertEqual(df_with_target.iloc[0]['target'], 0)
    
    def test_debe_seleccionar_features_correctas(self):
        """Test que valida selección de features"""
        from ml.feature_engineer import select_features
        feature_cols, target_col = select_features(self.df)
        self.assertIsInstance(feature_cols, list)
        self.assertGreater(len(feature_cols), 0)
        self.assertEqual(target_col, 'target')
    
    def test_debe_eliminar_ultima_fila_sin_target(self):
        """Test que valida que no se puede crear target para última fila"""
        from ml.feature_engineer import create_target
        df_with_target = create_target(self.df.copy())
        self.assertEqual(len(df_with_target), len(self.df) - 1)
    
    def test_debe_normalizar_features(self):
        """Test que valida normalización de features"""
        from ml.feature_engineer import normalize_features
        features = pd.DataFrame({
            'price': [100, 200, 300],
            'volume': [1000, 2000, 3000]
        })
        normalized = normalize_features(features)
        self.assertTrue((normalized >= 0).all().all())
        self.assertTrue((normalized <= 1).all().all())

if __name__ == '__main__':
    unittest.main()

