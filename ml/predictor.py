import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import select_features, normalize_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path):
    """
    Carga modelo entrenado desde disco.
    
    Args:
        model_path: Path al archivo .pkl del modelo
    
    Returns:
        Modelo cargado
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_price_direction(model, X):
    """
    Predice dirección del precio (0=baja, 1=sube).
    
    Args:
        model: Modelo entrenado
        X: Features (DataFrame o array)
    
    Returns:
        Predicción (0 o 1)
    """
    prediction = model.predict(X)[0]
    return int(prediction)

def predict_with_probability(model, X):
    """
    Predice dirección con probabilidad de confianza.
    
    Args:
        model: Modelo entrenado
        X: Features (DataFrame o array)
    
    Returns:
        Tuple (predicción, probabilidad)
    """
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    probability = float(probabilities[prediction])
    return int(prediction), probability

def interpret_prediction(prediction, probability):
    """
    Interpreta predicción en lenguaje natural.
    
    Args:
        prediction: 0 o 1
        probability: Probabilidad entre 0 y 1
    
    Returns:
        String con interpretación
    """
    direction = "SUBE" if prediction == 1 else "BAJA"
    confidence = f"{probability*100:.1f}%"
    return f"Predicción: El precio {direction} (Confianza: {confidence})"

def get_latest_model():
    """
    Obtiene el modelo más reciente de la carpeta models/.
    
    Returns:
        Path al modelo más reciente
    """
    models_dir = Path('models')
    models = list(models_dir.glob('bitcoin_predictor_*.pkl'))
    if not models:
        raise FileNotFoundError("No hay modelos entrenados. Ejecuta: python ml/train_bitcoin_predictor.py")
    latest_model = max(models, key=lambda x: x.stat().st_mtime)
    return latest_model

def predict_bitcoin_next_hour():
    """
    Predice la dirección del precio de Bitcoin en la próxima hora.
    
    Returns:
        Dict con predicción, dirección, probabilidad y timestamp
    """
    model_path = get_latest_model()
    logger.info(f"Usando modelo: {model_path.name}")
    
    model = load_trained_model(model_path)
    
    df = load_crypto_data(crypto_id='bitcoin', days=7)
    
    if len(df) == 0:
        raise ValueError("No hay datos disponibles para Bitcoin")
    
    latest_data = df.iloc[-1:]
    
    feature_cols, _ = select_features(df)
    X = latest_data[feature_cols]
    X_normalized = normalize_features(X)
    
    prediction, probability = predict_with_probability(model, X_normalized)
    interpretation = interpret_prediction(prediction, probability)
    
    return {
        'prediction': prediction,
        'direction': 'UP' if prediction == 1 else 'DOWN',
        'probability': probability,
        'confidence_pct': f"{probability*100:.2f}%",
        'interpretation': interpretation,
        'timestamp': datetime.now().isoformat(),
        'model_used': model_path.name
    }

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("BITCOIN PRICE PREDICTOR - Próxima Hora")
    logger.info("="*60)
    
    try:
        result = predict_bitcoin_next_hour()
        
        logger.info("\n" + "="*60)
        logger.info("PREDICCIÓN COMPLETADA")
        logger.info("="*60)
        logger.info(f"\nDirección: {result['direction']}")
        logger.info(f"Confianza: {result['confidence_pct']}")
        logger.info(f"\n{result['interpretation']}")
        logger.info(f"\nModelo usado: {result['model_used']}")
        logger.info(f"Timestamp: {result['timestamp']}")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Ejecuta primero: python ml/train_bitcoin_predictor.py")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")

