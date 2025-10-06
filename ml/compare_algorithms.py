import sys
import logging
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target, select_features, normalize_features
from ml.model_trainer import split_train_test
from ml.model_evaluator import generate_evaluation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_ml_algorithms(crypto_id='bitcoin', days=7):
    """
    Compara múltiples algoritmos de ML con los mismos datos.
    
    Returns:
        DataFrame con resultados comparativos
    """
    logger.info(f"Cargando datos de {crypto_id}...")
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    df_with_target = create_target(df)
    
    feature_cols, target_col = select_features(df_with_target)
    X = df_with_target[feature_cols]
    y = df_with_target[target_col]
    X_normalized = normalize_features(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y, test_size=0.2)
    
    algorithms = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    
    for name, model in algorithms.items():
        logger.info(f"\nEntrenando {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            evaluation = generate_evaluation_report(y_test, y_pred)
            
            results.append({
                'Algorithm': name,
                'Accuracy': evaluation['accuracy'],
                'Precision': evaluation['precision'],
                'Recall': evaluation['recall'],
                'F1-Score': evaluation['f1_score']
            })
            
            logger.info(f"{name} - Accuracy: {evaluation['accuracy']:.2%}")
        except Exception as e:
            logger.error(f"Error con {name}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    return results_df

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("COMPARACIÓN DE ALGORITMOS ML")
    logger.info("="*60)
    
    results = compare_ml_algorithms()
    
    print("\n" + "="*60)
    print("RESULTADOS:")
    print("="*60)
    print(results.to_string(index=False))
    print("="*60)
    
    best_algo = results.iloc[0]['Algorithm']
    best_acc = results.iloc[0]['Accuracy']
    print(f"\nMejor algoritmo: {best_algo} ({best_acc:.2%} accuracy)")

