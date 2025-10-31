import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Divide datos en entrenamiento y prueba.
    
    Args:
        X: Features
        y: Target
        test_size: Proporción de test (default 0.2)
        random_state: Semilla aleatoria (default 42)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def check_class_balance(y):
    """
    Verifica balance de clases en el target.
    
    Args:
        y: Array con variable target
    
    Returns:
        Dict con conteo por clase
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entrena modelo Random Forest para clasificación binaria.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de árboles (default 100)
        random_state: Semilla aleatoria (default 42)
    
    Returns:
        Modelo entrenado
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=20
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, model_name):
    """
    Guarda modelo entrenado en disco.
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del archivo (sin extensión)
    
    Returns:
        Path del archivo guardado
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    filepath = models_dir / f"{model_name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return str(filepath)

def train_naive_bayes(X_train, y_train):
    """
    Entrena modelo Naive Bayes para clasificación binaria.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        Modelo Naive Bayes entrenado
    """
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def load_model(model_path):
    """
    Carga modelo desde disco.
    
    Args:
        model_path: Path al archivo del modelo
    
    Returns:
        Modelo cargado
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

