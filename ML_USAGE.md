# 🤖 Sistema de Machine Learning - Bitcoin Price Predictor

## 📖 Descripción

Sistema completo de **predicción de precios de Bitcoin** que utiliza **Random Forest** para predecir si el precio subirá o bajará en la próxima hora.

**Tipo:** Clasificación binaria (UP=1, DOWN=0)  
**Algoritmo:** Random Forest (100 árboles)  
**Metodología:** Test-Driven Development (TDD)  
**Tests:** 27 tests pasando (100%)

---

## 🎯 Objetivo del Modelo

Predecir la **dirección del precio de Bitcoin** en la próxima hora basándose en:
- Precios históricos (lags 1h, 3h, 6h)
- Medias móviles (6h, 12h, 24h)
- Cambios porcentuales (1h, 3h)
- Indicadores técnicos (RSI 14h, 24h)
- Volatilidad y momentum

---

## 🚀 Uso Rápido

### 1. Hacer una predicción (usar modelo existente)

```bash
python ml/predictor.py
```

**Salida esperada:**
```
============================================================
PREDICCIÓN COMPLETADA
============================================================

Dirección: UP ↗️
Confianza: 57.78%

Predicción: El precio SUBE (Confianza: 57.8%)

Modelo usado: bitcoin_predictor_20251006_183658.pkl
Timestamp: 2025-10-06T18:41:57
============================================================
```

---

### 2. Entrenar un nuevo modelo (con datos actualizados)

```bash
python ml/train_bitcoin_predictor.py
```

**Salida esperada:**
```
============================================================
BITCOIN PRICE DIRECTION PREDICTOR
============================================================
Iniciando entrenamiento para bitcoin
Usando 7 días de datos históricos
Datos cargados: 13 registros
...
Métricas finales:
  Accuracy:  66.67%
  Precision: 66.67%
  Recall:    100.00%
  F1-Score:  80.00%
```

---

### 3. Ejecutar tests

```bash
# Tests ML (27 tests)
pytest test/test_ml_*.py -v

# Todos los tests del proyecto (62 tests)
pytest test/ -v
```

---

## 📊 Estructura del Sistema ML

```
ml/
├── data_loader.py              # Carga datos de BigQuery
├── feature_engineer.py         # Prepara features para ML
├── model_trainer.py            # Entrena Random Forest
├── model_evaluator.py          # Evalúa métricas
├── predictor.py                # Hace predicciones
└── train_bitcoin_predictor.py # Script principal de entrenamiento

test/
├── test_ml_data_loader.py      # 5 tests
├── test_ml_feature_engineer.py # 6 tests
├── test_ml_model_trainer.py    # 5 tests
├── test_ml_model_evaluator.py  # 6 tests
└── test_ml_predictor.py        # 5 tests

models/
└── bitcoin_predictor_YYYYMMDD_HHMMSS.pkl  # Modelos entrenados
```

---

## 🔧 Uso Programático (Python)

### Cargar modelo y hacer predicción

```python
from ml.predictor import predict_bitcoin_next_hour

# Predicción completa
result = predict_bitcoin_next_hour()

print(f"Dirección: {result['direction']}")          # 'UP' o 'DOWN'
print(f"Confianza: {result['confidence_pct']}")     # '57.78%'
print(f"Interpretación: {result['interpretation']}") # Texto legible
```

### Entrenar modelo personalizado

```python
from ml.train_bitcoin_predictor import train_bitcoin_price_predictor

# Entrenar con datos de 14 días
result = train_bitcoin_price_predictor(crypto_id='bitcoin', days=14)

print(f"Accuracy: {result['evaluation']['accuracy']:.2%}")
print(f"Modelo guardado en: {result['model_path']}")
```

### Usar módulos individuales

```python
# 1. Cargar datos
from ml.data_loader import load_crypto_data
df = load_crypto_data(crypto_id='bitcoin', days=7)

# 2. Crear target
from ml.feature_engineer import create_target, select_features, normalize_features
df_with_target = create_target(df)

# 3. Seleccionar features
feature_cols, target_col = select_features(df_with_target)
X = df_with_target[feature_cols]
y = df_with_target[target_col]

# 4. Normalizar
X_normalized = normalize_features(X)

# 5. Entrenar
from ml.model_trainer import train_model, split_train_test
X_train, X_test, y_train, y_test = split_train_test(X_normalized, y)
model = train_model(X_train, y_train)

# 6. Evaluar
from ml.model_evaluator import generate_evaluation_report
y_pred = model.predict(X_test)
report = generate_evaluation_report(y_test, y_pred)
print(f"Accuracy: {report['accuracy']:.2%}")

# 7. Guardar
from ml.model_trainer import save_model
save_model(model, 'mi_modelo_bitcoin')
```

---

## 📈 Features Utilizadas (12)

| Feature | Descripción |
|---------|-------------|
| `price_lag_1h` | Precio hace 1 hora |
| `price_lag_3h` | Precio hace 3 horas |
| `price_lag_6h` | Precio hace 6 horas |
| `price_ma_6h` | Media móvil 6 horas |
| `price_ma_12h` | Media móvil 12 horas |
| `price_ma_24h` | Media móvil 24 horas |
| `price_change_pct_1h` | Cambio % en 1 hora |
| `price_change_pct_3h` | Cambio % en 3 horas |
| `rsi_14h` | RSI 14 horas |
| `rsi_24h` | RSI 24 horas |
| `volatility_24h` | Volatilidad 24 horas |
| `price_momentum_6h` | Momentum 6 horas |

---

## ⚙️ Configuración del Modelo

**Random Forest:**
- `n_estimators=100` (100 árboles)
- `max_depth=10` (profundidad máxima)
- `min_samples_split=20` (mínimo para split)
- `random_state=42` (reproducibilidad)

**Train/Test Split:**
- 80% entrenamiento
- 20% prueba
- Estratificado por clase

**Normalización:**
- MinMaxScaler (0-1)

---

## 📊 Interpretación de Resultados

### Métricas

- **Accuracy:** % de predicciones correctas
- **Precision:** De las predicciones UP, cuántas fueron correctas
- **Recall:** De todas las subidas reales, cuántas se detectaron
- **F1-Score:** Media armónica de Precision y Recall

### Niveles de Confianza

| Confianza | Interpretación |
|-----------|----------------|
| > 70% | Alta confianza |
| 60-70% | Confianza moderada |
| 50-60% | Baja confianza |
| < 50% | Modelo incierto |

---

## ⚠️ Limitaciones Actuales

### Datos Insuficientes

**Actual:** ~13 registros de Bitcoin  
**Mínimo recomendado:** 30+ registros  
**Óptimo:** 100+ registros

Con más datos (acumulados por GitHub Actions cada 6 horas):
- **1 semana** = ~28 snapshots → Mejora significativa
- **2 semanas** = ~56 snapshots → Confianza >70%
- **1 mes** = ~120 snapshots → Predicciones robustas

### Advertencias

⚠️ Este modelo es **educativo/experimental**  
⚠️ NO usar para decisiones financieras reales  
⚠️ El mercado cripto es altamente volátil  
⚠️ Resultados pasados NO garantizan resultados futuros  

---

## 🔄 Workflow Recomendado

### Cada semana:

1. **Verificar datos acumulados:**
   ```bash
   python scripts/monitoring_dashboard.py
   ```

2. **Si hay >30 snapshots, reentrenar:**
   ```bash
   python ml/train_bitcoin_predictor.py
   ```

3. **Hacer nuevas predicciones:**
   ```bash
   python ml/predictor.py
   ```

4. **Validar métricas:**
   - Accuracy debería mejorar con más datos
   - F1-Score >75% indica buen rendimiento

---

## 🧪 Tests Implementados (27 tests)

### `test_ml_data_loader.py` (5 tests)
- ✅ Carga datos de BigQuery
- ✅ Elimina nulls en features críticas
- ✅ Ordena por fecha ascendente
- ✅ Tiene columnas requeridas
- ✅ Valida crypto_id obligatorio

### `test_ml_feature_engineer.py` (6 tests)
- ✅ Crea target binario (0/1)
- ✅ Elimina última fila sin target
- ✅ Normaliza features (0-1)
- ✅ Selecciona features correctas
- ✅ Target=0 cuando precio baja
- ✅ Target=1 cuando precio sube

### `test_ml_model_trainer.py` (5 tests)
- ✅ Divide train/test correctamente
- ✅ Entrena modelo Random Forest
- ✅ Guarda modelo en disco
- ✅ Valida balance de clases
- ✅ Modelo hace predicciones válidas

### `test_ml_model_evaluator.py` (6 tests)
- ✅ Calcula accuracy
- ✅ Calcula precision
- ✅ Calcula recall
- ✅ Calcula F1-score
- ✅ Genera matriz de confusión
- ✅ Genera reporte completo

### `test_ml_predictor.py` (5 tests)
- ✅ Carga modelo desde disco
- ✅ Predice dirección (0/1)
- ✅ Retorna probabilidades
- ✅ Predicción completa de Bitcoin
- ✅ Interpreta predicción correctamente

---

## 🎓 Próximos Pasos (Mejoras Futuras)

### Corto plazo
- ✅ Esperar acumulación de datos (2-3 semanas)
- ✅ Reentrenar con más snapshots
- ✅ Validar mejora de métricas

### Mediano plazo
- 🔲 Probar con otras cryptos (Ethereum, Solana)
- 🔲 Implementar cross-validation
- 🔲 Agregar más features (volumen, market cap)
- 🔲 Probar otros algoritmos (XGBoost, LightGBM)

### Largo plazo
- 🔲 Implementar LSTM para secuencias temporales
- 🔲 Crear ensemble de modelos
- 🔲 API REST para predicciones
- 🔲 Dashboard web interactivo

---

## 📚 Referencias

- **Algoritmo:** Random Forest (scikit-learn)
- **Metodología:** TDD (Test-Driven Development)
- **Arquitectura:** Ver `Arquitecture.md`
- **ETL Pipeline:** Ver `README.md`

---

**Sistema desarrollado siguiendo TDD estricto** 🔴→✅  
**27 tests escritos ANTES del código**  
**100% de tests pasando**

