import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestTemporalFeatures(unittest.TestCase):
    
    def setUp(self):
        dates = pd.date_range('2025-10-01', periods=48, freq='h')
        self.df = pd.DataFrame({
            'id': ['bitcoin'] * 48,
            'last_updated': dates,
            'current_price': [62000 + i * 100 for i in range(48)],
            'total_volume': [1000000000 + i * 1000000 for i in range(48)],
        })
        self.df = self.df.sort_values('last_updated')
    
    def test_should_calculate_price_lag_1h(self):
        df = self.df.copy()
        df['price_lag_1h'] = df.groupby('id')['current_price'].shift(1)
        self.assertTrue('price_lag_1h' in df.columns)
        self.assertTrue(pd.isna(df.iloc[0]['price_lag_1h']))
        self.assertEqual(df.iloc[1]['price_lag_1h'], df.iloc[0]['current_price'])
    
    def test_should_calculate_price_lag_24h(self):
        df = self.df.copy()
        df['price_lag_24h'] = df.groupby('id')['current_price'].shift(24)
        self.assertTrue('price_lag_24h' in df.columns)
        for i in range(24):
            self.assertTrue(pd.isna(df.iloc[i]['price_lag_24h']))
        self.assertEqual(df.iloc[24]['price_lag_24h'], df.iloc[0]['current_price'])
    
    def test_should_calculate_rolling_mean_24h(self):
        df = self.df.copy()
        df['price_ma_24h'] = df.groupby('id')['current_price'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
        self.assertTrue('price_ma_24h' in df.columns)
        self.assertGreater(df.iloc[-1]['price_ma_24h'], df.iloc[0]['current_price'])
    
    def test_should_calculate_rolling_std(self):
        df = self.df.copy()
        df['price_std_24h'] = df.groupby('id')['current_price'].transform(lambda x: x.rolling(window=24, min_periods=2).std())
        self.assertTrue('price_std_24h' in df.columns)
        self.assertGreater(df.iloc[-1]['price_std_24h'], 0)
    
    def test_should_calculate_price_change_percent_1h(self):
        df = self.df.copy()
        df['price_lag_1h'] = df.groupby('id')['current_price'].shift(1)
        df['price_change_pct_1h'] = ((df['current_price'] - df['price_lag_1h']) / df['price_lag_1h']) * 100
        self.assertTrue('price_change_pct_1h' in df.columns)
        self.assertTrue(pd.isna(df.iloc[0]['price_change_pct_1h']))
        self.assertAlmostEqual(df.iloc[1]['price_change_pct_1h'], 0.161, places=2)
    
    def test_should_calculate_volume_change_24h(self):
        df = self.df.copy()
        df['volume_lag_24h'] = df.groupby('id')['total_volume'].shift(24)
        df['volume_change_24h'] = df['total_volume'] - df['volume_lag_24h']
        self.assertTrue('volume_change_24h' in df.columns)
        self.assertEqual(df.iloc[24]['volume_change_24h'], 24 * 1000000)
    
    def test_should_calculate_momentum_indicators(self):
        df = self.df.copy()
        df['price_momentum_6h'] = df.groupby('id')['current_price'].transform(lambda x: x.diff(6))
        self.assertTrue('price_momentum_6h' in df.columns)
        self.assertEqual(df.iloc[6]['price_momentum_6h'], 600)
    
    def test_should_handle_multiple_cryptos(self):
        dates = pd.date_range('2025-10-01', periods=24, freq='h')
        df_multi = pd.DataFrame({
            'id': ['bitcoin'] * 24 + ['ethereum'] * 24,
            'last_updated': dates.tolist() + dates.tolist(),
            'current_price': [62000 + i * 100 for i in range(24)] + [2500 + i * 10 for i in range(24)],
        })
        df_multi = df_multi.sort_values(['id', 'last_updated'])
        df_multi['price_lag_1h'] = df_multi.groupby('id')['current_price'].shift(1)
        bitcoin_data = df_multi[df_multi['id'] == 'bitcoin']
        ethereum_data = df_multi[df_multi['id'] == 'ethereum']
        self.assertEqual(len(bitcoin_data), 24)
        self.assertEqual(len(ethereum_data), 24)
        self.assertTrue(pd.isna(bitcoin_data.iloc[0]['price_lag_1h']))
        self.assertTrue(pd.isna(ethereum_data.iloc[0]['price_lag_1h']))
    
    def test_should_calculate_rsi_components(self):
        df = self.df.copy()
        df['price_change'] = df.groupby('id')['current_price'].diff()
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        self.assertTrue('gain' in df.columns)
        self.assertTrue('loss' in df.columns)
        self.assertTrue((df['gain'] >= 0).all())
        self.assertTrue((df['loss'] >= 0).all())
    
    def test_should_validate_temporal_ordering(self):
        df = self.df.copy()
        df_sorted = df.sort_values(['id', 'last_updated'])
        is_sorted = (df_sorted['last_updated'].diff()[1:] >= pd.Timedelta(0)).all()
        self.assertTrue(is_sorted, "Los datos deben estar ordenados temporalmente")

class TestDataCompleteness(unittest.TestCase):
    
    def test_should_detect_time_gaps(self):
        dates_with_gap = pd.date_range('2025-10-01', periods=24, freq='h').tolist()
        dates_with_gap = dates_with_gap[:10] + dates_with_gap[15:]
        df = pd.DataFrame({
            'id': ['bitcoin'] * len(dates_with_gap),
            'last_updated': dates_with_gap,
        })
        df = df.sort_values('last_updated')
        time_diffs = df['last_updated'].diff()
        expected_freq = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_freq * 1.5]
        self.assertGreater(len(gaps), 0, "Debe detectar gaps en la serie temporal")
    
    def test_should_validate_minimum_data_points(self):
        min_required = 24
        df_short = pd.DataFrame({
            'id': ['bitcoin'] * 10,
            'current_price': range(10),
        })
        self.assertLess(len(df_short), min_required, "Debe detectar series temporales muy cortas")
    
    def test_should_calculate_data_coverage_percentage(self):
        expected_hours = 48
        actual_records = 40
        coverage = (actual_records / expected_hours) * 100
        self.assertLess(coverage, 100, "Cobertura debe ser menor a 100% si faltan datos")
        self.assertAlmostEqual(coverage, 83.33, places=1)

if __name__ == '__main__':
    unittest.main()

