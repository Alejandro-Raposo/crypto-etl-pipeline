import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

class TestAdvancedFeatures(unittest.TestCase):
    
    def setUp(self):
        dates = pd.date_range('2025-10-01', periods=50, freq='h')
        np.random.seed(42)
        
        self.df = pd.DataFrame({
            'id': ['bitcoin'] * 50,
            'last_updated': dates,
            'current_price': 62000 + np.random.randn(50) * 1000,
            'total_volume': 1000000000 + np.random.randn(50) * 50000000,
            'high_24h': 63000 + np.random.randn(50) * 500,
            'low_24h': 61000 + np.random.randn(50) * 500,
        })
        
        for lag in [1, 6, 12, 24]:
            self.df[f'price_lag_{lag}h'] = self.df.groupby('id')['current_price'].shift(lag)
            
        for window in [6, 12, 24]:
            self.df[f'price_ma_{window}h'] = self.df.groupby('id')['current_price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'price_std_{window}h'] = self.df.groupby('id')['current_price'].transform(
                lambda x: x.rolling(window=window, min_periods=2).std()
            )
    
    def test_calcula_macd_correctamente(self):
        """Verifica que MACD se calcula correctamente"""
        from scripts.feature_engineering_temporal import calculate_macd_features
        
        df = calculate_macd_features(self.df.copy())
        
        self.assertIn('macd', df.columns)
        self.assertIn('macd_signal', df.columns)
        self.assertIn('macd_histogram', df.columns)
        
        non_null_macd = df['macd'].notna().sum()
        self.assertGreater(non_null_macd, 0, "Debe haber valores MACD calculados")
    
    def test_calcula_bollinger_bands(self):
        """Verifica que Bollinger Bands se calculan"""
        from scripts.feature_engineering_temporal import calculate_bollinger_bands
        
        df = calculate_bollinger_bands(self.df.copy())
        
        self.assertIn('bb_upper', df.columns)
        self.assertIn('bb_lower', df.columns)
        self.assertIn('bb_position', df.columns)
        
        non_null_bb = df['bb_position'].notna().sum()
        self.assertGreater(non_null_bb, 0, "Debe haber posiciones BB calculadas")
        
        valid_positions = df['bb_position'].dropna()
        self.assertTrue((valid_positions >= 0).all() and (valid_positions <= 1).all(),
                       "BB position debe estar entre 0 y 1")
    
    def test_calcula_volume_features(self):
        """Verifica que features de volumen se calculan"""
        from scripts.feature_engineering_temporal import calculate_volume_features
        
        df = calculate_volume_features(self.df.copy())
        
        self.assertIn('volume_ma_6h', df.columns)
        self.assertIn('volume_ma_24h', df.columns)
        self.assertIn('volume_ratio_6h_24h', df.columns)
        
        valid_ratios = df['volume_ratio_6h_24h'].dropna()
        self.assertTrue((valid_ratios > 0).all(), "Volume ratios deben ser positivos")
    
    def test_calcula_time_features(self):
        """Verifica que features temporales cíclicas se calculan"""
        from scripts.feature_engineering_temporal import calculate_time_features
        
        df = calculate_time_features(self.df.copy())
        
        self.assertIn('hour_sin', df.columns)
        self.assertIn('hour_cos', df.columns)
        self.assertIn('day_sin', df.columns)
        self.assertIn('day_cos', df.columns)
        
        self.assertTrue((df['hour_sin'] >= -1).all() and (df['hour_sin'] <= 1).all(),
                       "hour_sin debe estar entre -1 y 1")
        self.assertTrue((df['hour_cos'] >= -1).all() and (df['hour_cos'] <= 1).all(),
                       "hour_cos debe estar entre -1 y 1")
    
    def test_calcula_volatility_features(self):
        """Verifica que features de volatilidad avanzadas se calculan"""
        from scripts.feature_engineering_temporal import calculate_advanced_volatility
        
        df = calculate_advanced_volatility(self.df.copy())
        
        self.assertIn('atr_14h', df.columns)
        self.assertIn('volatility_ratio', df.columns)
        
        valid_atr = df['atr_14h'].dropna()
        self.assertTrue((valid_atr >= 0).all(), "ATR debe ser no negativo")
    
    def test_calcula_price_acceleration(self):
        """Verifica que aceleración de precio se calcula"""
        from scripts.feature_engineering_temporal import calculate_price_acceleration
        
        df = calculate_price_acceleration(self.df.copy())
        
        self.assertIn('price_acceleration_6h', df.columns)
        self.assertIn('roc_12h', df.columns)
        
        non_null_acc = df['price_acceleration_6h'].notna().sum()
        self.assertGreater(non_null_acc, 0, "Debe haber aceleraciones calculadas")
    
    def test_todas_features_juntas(self):
        """Verifica que todas las features se pueden calcular juntas"""
        from scripts.feature_engineering_temporal import add_advanced_features
        
        df = add_advanced_features(self.df.copy())
        
        initial_cols = len(self.df.columns)
        final_cols = len(df.columns)
        
        self.assertGreater(final_cols, initial_cols,
                          "Deben agregarse nuevas features")
        self.assertGreaterEqual(final_cols - initial_cols, 10,
                               "Deben agregarse al menos 10 nuevas features")

if __name__ == '__main__':
    unittest.main()

