import unittest
import numpy as np
from sklearn.metrics import accuracy_score

class TestMLModelEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Crear predicciones sintéticas para tests"""
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    
    def test_debe_calcular_accuracy(self):
        """Test que valida cálculo de accuracy"""
        from ml.model_evaluator import calculate_accuracy
        acc = calculate_accuracy(self.y_true, self.y_pred)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
    
    def test_debe_calcular_precision(self):
        """Test que valida cálculo de precision"""
        from ml.model_evaluator import calculate_precision
        prec = calculate_precision(self.y_true, self.y_pred)
        self.assertIsInstance(prec, float)
        self.assertGreaterEqual(prec, 0.0)
        self.assertLessEqual(prec, 1.0)
    
    def test_debe_calcular_recall(self):
        """Test que valida cálculo de recall"""
        from ml.model_evaluator import calculate_recall
        rec = calculate_recall(self.y_true, self.y_pred)
        self.assertIsInstance(rec, float)
        self.assertGreaterEqual(rec, 0.0)
        self.assertLessEqual(rec, 1.0)
    
    def test_debe_calcular_f1_score(self):
        """Test que valida cálculo de F1"""
        from ml.model_evaluator import calculate_f1_score
        f1 = calculate_f1_score(self.y_true, self.y_pred)
        self.assertIsInstance(f1, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
    
    def test_debe_generar_matriz_confusion(self):
        """Test que valida matriz de confusión"""
        from ml.model_evaluator import generate_confusion_matrix
        cm = generate_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.y_true))
    
    def test_debe_generar_reporte_completo(self):
        """Test que valida reporte de métricas"""
        from ml.model_evaluator import generate_evaluation_report
        report = generate_evaluation_report(self.y_true, self.y_pred)
        self.assertIsInstance(report, dict)
        self.assertIn('accuracy', report)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1_score', report)

if __name__ == '__main__':
    unittest.main()

