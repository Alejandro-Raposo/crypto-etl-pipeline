import unittest
import pandas as pd
import numpy as np
from pathlib import Path

class TestMLPredictor(unittest.TestCase):
    
    def test_debe_cargar_modelo_existente(self):
        """Test que valida carga de modelo desde disco"""
        from ml.predictor import load_trained_model
        from ml.model_trainer import train_model, save_model
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = np.random.randint(0, 2, 50)
        model = train_model(X_train, y_train)
        model_path = save_model(model, 'test_predictor_model')
        loaded_model = load_trained_model(model_path)
        self.assertIsNotNone(loaded_model)
        Path(model_path).unlink()
    
    def test_debe_predecir_direccion_precio(self):
        """Test que valida predicción básica"""
        from ml.predictor import predict_price_direction
        from ml.model_trainer import train_model
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = np.random.randint(0, 2, 50)
        model = train_model(X_train, y_train)
        X_new = pd.DataFrame(np.random.randn(1, 3))
        prediction = predict_price_direction(model, X_new)
        self.assertIn(prediction, [0, 1])
    
    def test_debe_retornar_probabilidades(self):
        """Test que valida predicción con probabilidades"""
        from ml.predictor import predict_with_probability
        from ml.model_trainer import train_model
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = np.random.randint(0, 2, 50)
        model = train_model(X_train, y_train)
        X_new = pd.DataFrame(np.random.randn(1, 3))
        prediction, probability = predict_with_probability(model, X_new)
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_debe_predecir_bitcoin_actual(self):
        """Test de integración: predicción completa de Bitcoin"""
        from ml.predictor import predict_bitcoin_next_hour
        try:
            result = predict_bitcoin_next_hour()
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('direction', result)
            self.assertIn('probability', result)
        except FileNotFoundError:
            self.skipTest("No hay modelo entrenado disponible")
    
    def test_debe_interpretar_prediccion_correctamente(self):
        """Test que valida interpretación de resultados"""
        from ml.predictor import interpret_prediction
        interpretation_up = interpret_prediction(1, 0.85)
        interpretation_down = interpret_prediction(0, 0.72)
        self.assertIn('SUBE', interpretation_up)
        self.assertIn('85', interpretation_up)
        self.assertIn('BAJA', interpretation_down)
        self.assertIn('72', interpretation_down)

if __name__ == '__main__':
    unittest.main()

