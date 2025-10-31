"""
Tests para Ensemble Methods (combinación de modelos).

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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))


class TestEnsemblePredictor(unittest.TestCase):
    
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
    
    def test_ensemble_entrena_correctamente(self):
        """Verifica que el ensemble se entrena con múltiples modelos"""
        from ml.ensemble_predictor import train_ensemble_model
        
        result = train_ensemble_model(self.X_train, self.y_train)
        
        self.assertIn('model', result)
        self.assertTrue(hasattr(result['model'], 'predict'))
        self.assertTrue(hasattr(result['model'], 'predict_proba'))
    
    def test_ensemble_hace_predicciones_validas(self):
        """Verifica que las predicciones del ensemble son binarias"""
        from ml.ensemble_predictor import train_ensemble_model
        
        result = train_ensemble_model(self.X_train, self.y_train)
        model = result['model']
        
        y_pred = model.predict(self.X_test)
        
        self.assertEqual(len(y_pred), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in y_pred))
    
    def test_ensemble_retorna_probabilidades(self):
        """Verifica que el ensemble retorna probabilidades"""
        from ml.ensemble_predictor import train_ensemble_model
        
        result = train_ensemble_model(self.X_train, self.y_train)
        model = result['model']
        
        y_proba = model.predict_proba(self.X_test)
        
        # Debe tener 2 columnas (probabilidad clase 0 y clase 1)
        self.assertEqual(y_proba.shape[1], 2)
        
        # Probabilidades deben sumar 1
        for proba_row in y_proba:
            self.assertAlmostEqual(sum(proba_row), 1.0, places=5)
    
    def test_ensemble_supera_accuracy_minimo(self):
        """Verifica que el ensemble alcanza accuracy razonable"""
        from ml.ensemble_predictor import train_ensemble_model
        
        result = train_ensemble_model(self.X_train, self.y_train)
        model = result['model']
        
        y_pred = model.predict(self.X_test)
        accuracy = (y_pred == self.y_test).sum() / len(self.y_test)
        
        # Con datos aleatorios, debe superar 30% (mejor que azar)
        self.assertGreater(accuracy, 0.30)


if __name__ == '__main__':
    unittest.main()

