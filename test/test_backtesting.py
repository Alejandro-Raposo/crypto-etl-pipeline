"""
Tests para backtesting temporal de predicciones.

Siguiendo TDD estricto según Arquitecture.md:
- Tests PRIMERO, implementación DESPUÉS
- Mínimo 5 tests por módulo
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent))


class TestBacktesting(unittest.TestCase):
    
    def setUp(self):
        """Prepara datos sintéticos para tests"""
        np.random.seed(42)
        n_samples = 100
        
        dates = pd.date_range('2025-10-01', periods=n_samples, freq='h')
        
        self.df = pd.DataFrame({
            'last_updated': dates,
            'current_price': 60000 + np.random.randn(n_samples) * 100,
            'price_lag_1h': 60000 + np.random.randn(n_samples) * 100,
            'price_ma_6h': 60000 + np.random.randn(n_samples) * 50,
            'rsi_14h': np.random.uniform(20, 80, n_samples),
            'volatility_24h': np.random.uniform(0.01, 0.05, n_samples),
        })
        
        self.df['target'] = (self.df['current_price'] > self.df['current_price'].shift(1)).astype(int)
        self.df = self.df.dropna()
        
        self.feature_cols = ['price_lag_1h', 'price_ma_6h', 'rsi_14h', 'volatility_24h']
        self.target_col = 'target'
        
        # Entrenar modelo simple
        X = self.df[self.feature_cols].iloc[:50]
        y = self.df[self.target_col].iloc[:50]
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
    
    def test_backtest_retorna_accuracy(self):
        """Verifica que backtesting calcula accuracy"""
        from ml.backtesting import backtest_predictions
        
        result = backtest_predictions(
            self.model, 
            self.df, 
            self.feature_cols, 
            self.target_col, 
            window=24
        )
        
        self.assertIn('accuracy', result)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
    
    def test_retorna_dataframe_resultados(self):
        """Verifica que retorna DataFrame con resultados"""
        from ml.backtesting import backtest_predictions
        
        result = backtest_predictions(
            self.model, 
            self.df, 
            self.feature_cols, 
            self.target_col, 
            window=24
        )
        
        self.assertIn('results', result)
        self.assertIsInstance(result['results'], pd.DataFrame)
        self.assertIn('predicted', result['results'].columns)
        self.assertIn('actual', result['results'].columns)
        self.assertIn('correct', result['results'].columns)
    
    def test_no_usa_datos_futuros(self):
        """Verifica que no hay data leakage temporal"""
        from ml.backtesting import backtest_predictions
        
        window = 24
        result = backtest_predictions(
            self.model, 
            self.df, 
            self.feature_cols, 
            self.target_col, 
            window=window
        )
        
        # Solo debe haber predicciones después de la ventana inicial
        expected_predictions = len(self.df) - window
        self.assertEqual(len(result['results']), expected_predictions)
    
    def test_predicciones_son_binarias(self):
        """Verifica que todas las predicciones son 0 o 1"""
        from ml.backtesting import backtest_predictions
        
        result = backtest_predictions(
            self.model, 
            self.df, 
            self.feature_cols, 
            self.target_col, 
            window=24
        )
        
        predictions = result['results']['predicted'].values
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_cuenta_correctamente_predicciones_correctas(self):
        """Verifica que cuenta correctamente las predicciones acertadas"""
        from ml.backtesting import backtest_predictions
        
        result = backtest_predictions(
            self.model, 
            self.df, 
            self.feature_cols, 
            self.target_col, 
            window=24
        )
        
        expected_correct = result['results']['correct'].sum()
        
        self.assertIn('correct_predictions', result)
        self.assertEqual(result['correct_predictions'], expected_correct)
        
        # Accuracy debe ser correcto
        expected_accuracy = expected_correct / len(result['results'])
        self.assertAlmostEqual(result['accuracy'], expected_accuracy, places=6)


if __name__ == '__main__':
    unittest.main()

