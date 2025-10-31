"""
Tests para optimización de hiperparámetros con Grid Search.

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
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))


class TestHyperparameterTuning(unittest.TestCase):
    
    def setUp(self):
        """Prepara datos sintéticos para tests"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    def test_grid_search_retorna_mejor_modelo(self):
        """Verifica que Grid Search retorna un modelo entrenado"""
        from ml.hyperparameter_tuning import optimize_random_forest
        
        result = optimize_random_forest(self.X_train, self.y_train)
        
        self.assertIn('best_model', result)
        self.assertIsInstance(result['best_model'], RandomForestClassifier)
        self.assertTrue(hasattr(result['best_model'], 'predict'))
    
    def test_retorna_mejores_parametros(self):
        """Verifica que retorna los mejores parámetros encontrados"""
        from ml.hyperparameter_tuning import optimize_random_forest
        
        result = optimize_random_forest(self.X_train, self.y_train)
        
        self.assertIn('best_params', result)
        self.assertIn('n_estimators', result['best_params'])
        self.assertIn('max_depth', result['best_params'])
        self.assertIsInstance(result['best_params']['n_estimators'], int)
    
    def test_best_score_es_valido(self):
        """Verifica que el mejor score está en rango válido"""
        from ml.hyperparameter_tuning import optimize_random_forest
        
        result = optimize_random_forest(self.X_train, self.y_train)
        
        self.assertIn('best_score', result)
        self.assertGreaterEqual(result['best_score'], 0.0)
        self.assertLessEqual(result['best_score'], 1.0)
    
    def test_modelo_optimizado_hace_predicciones(self):
        """Verifica que el modelo optimizado puede predecir"""
        from ml.hyperparameter_tuning import optimize_random_forest
        
        result = optimize_random_forest(self.X_train, self.y_train)
        model = result['best_model']
        
        y_pred = model.predict(self.X_test)
        
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertTrue(all(pred in [0, 1] for pred in y_pred))


if __name__ == '__main__':
    unittest.main()

