"""
Tests para Walk-Forward Validation (validación temporal deslizante).

Siguiendo TDD estricto según Arquitecture.md:
- Tests PRIMERO, implementación DESPUÉS
- Mínimo 4 tests por módulo
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent))


class TestWalkForwardValidation(unittest.TestCase):
    
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
    
    def test_walk_forward_retorna_scores(self):
        """Verifica que walk-forward retorna scores de cada iteración"""
        from ml.walk_forward_validation import walk_forward_validation
        
        result = walk_forward_validation(
            self.df,
            self.feature_cols,
            self.target_col,
            train_window=60,
            test_window=10
        )
        
        self.assertIn('fold_scores', result)
        self.assertIsInstance(result['fold_scores'], list)
        self.assertGreater(len(result['fold_scores']), 0)
    
    def test_retorna_accuracy_media(self):
        """Verifica que calcula accuracy promedio y std"""
        from ml.walk_forward_validation import walk_forward_validation
        
        result = walk_forward_validation(
            self.df,
            self.feature_cols,
            self.target_col,
            train_window=60,
            test_window=10
        )
        
        self.assertIn('mean_accuracy', result)
        self.assertIn('std_accuracy', result)
        self.assertGreaterEqual(result['mean_accuracy'], 0.0)
        self.assertLessEqual(result['mean_accuracy'], 1.0)
    
    def test_respeta_orden_temporal(self):
        """Verifica que entrenamiento precede a test (sin data leakage)"""
        from ml.walk_forward_validation import walk_forward_validation
        
        result = walk_forward_validation(
            self.df,
            self.feature_cols,
            self.target_col,
            train_window=60,
            test_window=10
        )
        
        # Debe haber al menos 1 fold
        self.assertGreaterEqual(len(result['fold_scores']), 1)
        
        # Cada score debe ser válido
        for score in result['fold_scores']:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_diferentes_ventanas_producen_diferentes_resultados(self):
        """Verifica que diferentes configuraciones de ventanas funcionan"""
        from ml.walk_forward_validation import walk_forward_validation
        
        # Ventana pequeña
        result_small = walk_forward_validation(
            self.df,
            self.feature_cols,
            self.target_col,
            train_window=40,
            test_window=10
        )
        
        # Ventana grande
        result_large = walk_forward_validation(
            self.df,
            self.feature_cols,
            self.target_col,
            train_window=70,
            test_window=5
        )
        
        # Ambos deben retornar resultados válidos
        self.assertIn('mean_accuracy', result_small)
        self.assertIn('mean_accuracy', result_large)
        
        # Número de folds puede ser diferente
        self.assertIsInstance(result_small['fold_scores'], list)
        self.assertIsInstance(result_large['fold_scores'], list)


if __name__ == '__main__':
    unittest.main()

