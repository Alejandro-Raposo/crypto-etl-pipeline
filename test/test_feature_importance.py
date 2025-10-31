import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

sys.path.append(str(Path(__file__).parent.parent))

class TestFeatureImportance(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        y_train = np.random.randint(0, 2, 100)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X_train, y_train)
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    def test_importancias_suman_100_pct(self):
        """Verifica que importancias suman 100%"""
        from ml.feature_importance import analyze_feature_importance
        
        result = analyze_feature_importance(self.model, self.feature_names)
        
        total_pct = result['importance_pct'].sum()
        self.assertLess(
            abs(total_pct - 100.0), 
            0.01, 
            "Importancias deben sumar 100%"
        )
    
    def test_ordenamiento_descendente(self):
        """Verifica ordenamiento por importancia"""
        from ml.feature_importance import analyze_feature_importance
        
        result = analyze_feature_importance(self.model, self.feature_names)
        
        self.assertTrue(
            result['importance'].is_monotonic_decreasing,
            "Importancias deben estar ordenadas descendentemente"
        )
    
    def test_retorna_dataframe(self):
        """Verifica que retorna DataFrame con columnas correctas"""
        from ml.feature_importance import analyze_feature_importance
        
        result = analyze_feature_importance(self.model, self.feature_names)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('feature', result.columns)
        self.assertIn('importance', result.columns)
        self.assertIn('rank', result.columns)
        self.assertIn('importance_pct', result.columns)
    
    def test_funciona_solo_con_modelos_tree_based(self):
        """Verifica que valida tipo de modelo"""
        from ml.feature_importance import analyze_feature_importance
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        y_train = np.random.randint(0, 2, 100)
        
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        
        with self.assertRaises(ValueError) as context:
            analyze_feature_importance(nb_model, ['feature1', 'feature2'])
        
        self.assertIn('feature_importances_', str(context.exception))
    
    def test_ranks_son_correctos(self):
        """Verifica que ranks van de 1 a N"""
        from ml.feature_importance import analyze_feature_importance
        
        result = analyze_feature_importance(self.model, self.feature_names)
        
        expected_ranks = list(range(1, len(self.feature_names) + 1))
        actual_ranks = result['rank'].tolist()
        
        self.assertEqual(actual_ranks, expected_ranks)

if __name__ == '__main__':
    unittest.main()

