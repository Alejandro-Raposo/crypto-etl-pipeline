import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

sys.path.append(str(Path(__file__).parent.parent))

class TestCrossValidation(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        self.y = np.random.randint(0, 2, 100)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    def test_cross_validation_retorna_5_scores(self):
        """Verifica que CV retorna 5 scores con cv=5"""
        from ml.cross_validation import evaluate_with_cross_validation
        
        result = evaluate_with_cross_validation(self.X, self.y, self.model, cv=5)
        
        self.assertIn('scores', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertEqual(len(result['scores']), 5)
        self.assertGreaterEqual(result['mean'], 0)
        self.assertLessEqual(result['mean'], 1)
    
    def test_std_no_excesiva(self):
        """Verifica que desviación estándar es razonable"""
        from ml.cross_validation import evaluate_with_cross_validation
        
        result = evaluate_with_cross_validation(self.X, self.y, self.model, cv=5)
        
        self.assertLess(result['std'], 0.3, "Desviación estándar no debe exceder 30%")
    
    def test_acepta_diferentes_cv_folds(self):
        """Verifica que acepta diferentes números de folds"""
        from ml.cross_validation import evaluate_with_cross_validation
        
        result_3 = evaluate_with_cross_validation(self.X, self.y, self.model, cv=3)
        self.assertEqual(len(result_3['scores']), 3)
        
        result_10 = evaluate_with_cross_validation(self.X, self.y, self.model, cv=10)
        self.assertEqual(len(result_10['scores']), 10)
    
    def test_funciona_con_naive_bayes(self):
        """Verifica compatibilidad con diferentes modelos"""
        from ml.cross_validation import evaluate_with_cross_validation
        
        nb_model = GaussianNB()
        result = evaluate_with_cross_validation(self.X, self.y, nb_model, cv=5)
        
        self.assertIn('mean', result)
        self.assertGreaterEqual(result['mean'], 0)
        self.assertLessEqual(result['mean'], 1)
    
    def test_retorna_min_max(self):
        """Verifica que retorna valores mínimo y máximo"""
        from ml.cross_validation import evaluate_with_cross_validation
        
        result = evaluate_with_cross_validation(self.X, self.y, self.model, cv=5)
        
        self.assertIn('min', result)
        self.assertIn('max', result)
        self.assertLessEqual(result['min'], result['mean'])
        self.assertGreaterEqual(result['max'], result['mean'])

if __name__ == '__main__':
    unittest.main()

