import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class TestMLModelTrainer(unittest.TestCase):
    
    def setUp(self):
        """Crear datos sintéticos para tests"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        self.y_train = np.random.randint(0, 2, 100)
        
        self.X_test = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'feature3': np.random.randn(30),
        })
        self.y_test = np.random.randint(0, 2, 30)
    
    def test_debe_entrenar_modelo_correctamente(self):
        """Test que valida entrenamiento básico"""
        from ml.model_trainer import train_model
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
    
    def test_modelo_debe_hacer_predicciones(self):
        """Test que valida capacidad de predicción"""
        from ml.model_trainer import train_model
        model = train_model(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_debe_dividir_train_test_correctamente(self):
        """Test que valida split de datos"""
        from ml.model_trainer import split_train_test
        X = pd.DataFrame(np.random.randn(100, 3))
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_debe_validar_balance_de_clases(self):
        """Test que valida distribución de clases"""
        from ml.model_trainer import check_class_balance
        y = np.array([0] * 45 + [1] * 55)
        balance = check_class_balance(y)
        self.assertIsInstance(balance, dict)
        self.assertIn(0, balance)
        self.assertIn(1, balance)
        self.assertEqual(balance[0] + balance[1], 100)
    
    def test_debe_guardar_modelo_entrenado(self):
        """Test que valida guardado de modelo"""
        from ml.model_trainer import train_model, save_model
        model = train_model(self.X_train, self.y_train)
        save_path = save_model(model, 'test_model')
        import os
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

if __name__ == '__main__':
    unittest.main()

