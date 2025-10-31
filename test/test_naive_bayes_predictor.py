import unittest
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class TestNaiveBayesPredictor(unittest.TestCase):
    
    def test_accuracy_mayor_55_pct(self):
        """Naive Bayes debe superar 55% accuracy"""
        from ml.train_bitcoin_naive_bayes import train_naive_bayes_predictor
        
        result = train_naive_bayes_predictor(crypto_id='bitcoin', days=30)
        
        self.assertGreaterEqual(
            result['evaluation']['accuracy'], 
            0.55, 
            "Naive Bayes debe tener accuracy >= 55%"
        )
    
    def test_f1_score_mayor_50_pct(self):
        """F1-Score debe ser superior a 50%"""
        from ml.train_bitcoin_naive_bayes import train_naive_bayes_predictor
        
        result = train_naive_bayes_predictor(crypto_id='bitcoin', days=30)
        
        self.assertGreaterEqual(
            result['evaluation']['f1_score'], 
            0.50, 
            "F1-Score debe ser >= 50%"
        )
    
    def test_retorna_estructura_correcta(self):
        """Verifica estructura del resultado"""
        from ml.train_bitcoin_naive_bayes import train_naive_bayes_predictor
        
        result = train_naive_bayes_predictor(crypto_id='bitcoin', days=30)
        
        self.assertIn('model', result)
        self.assertIn('model_path', result)
        self.assertIn('evaluation', result)
        self.assertIn('feature_cols', result)
        self.assertIn('balance', result)
        self.assertIn('train_size', result)
        self.assertIn('test_size', result)
    
    def test_guarda_modelo_correctamente(self):
        """Verifica que modelo se guarda"""
        from ml.train_bitcoin_naive_bayes import train_naive_bayes_predictor
        
        result = train_naive_bayes_predictor(crypto_id='bitcoin', days=30)
        model_path = result['model_path']
        
        self.assertTrue(os.path.exists(model_path), "Archivo de modelo debe existir")
        self.assertTrue(model_path.endswith('.pkl'), "Modelo debe ser archivo .pkl")
        
        os.remove(model_path)
    
    def test_modelo_hace_predicciones_validas(self):
        """Verifica que predicciones son 0 o 1"""
        from ml.train_bitcoin_naive_bayes import train_naive_bayes_predictor
        from ml.data_loader import load_crypto_data
        from ml.feature_engineer import select_features, normalize_features
        import numpy as np
        
        result = train_naive_bayes_predictor(crypto_id='bitcoin', days=30)
        model = result['model']
        
        df = load_crypto_data(crypto_id='bitcoin', days=30)
        feature_cols, _ = select_features(df)
        X_test = normalize_features(df[feature_cols].head(10))
        
        predictions = model.predict(X_test)
        
        self.assertTrue(np.all(np.isin(predictions, [0, 1])), "Predicciones deben ser 0 o 1")
        
        os.remove(result['model_path'])

if __name__ == '__main__':
    unittest.main()

