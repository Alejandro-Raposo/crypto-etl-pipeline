# 📚 Documentación - Crypto ETL Pipeline + ML Predictor

**Proyecto:** Sistema ETL automatizado con Machine Learning para predicción de precios de criptomonedas  
**Última actualización:** 2025-10-24

---

## 📖 Descripción General

Pipeline ETL automatizado que:
- Extrae datos de criptomonedas desde **CoinGecko API**
- Procesa y almacena en **BigQuery (GCP)**
- Genera **98 features temporales** para Machine Learning
- Entrena modelos para predecir movimientos de precios (UP/DOWN)
- Se ejecuta **automáticamente cada hora** via GitHub Actions

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐
│  CoinGecko API  │
└────────┬────────┘
         │ extract.py (cada 1h)
         ▼
┌─────────────────┐
│   data/raw/     │ JSON
└────────┬────────┘
         │ transform.py
         ▼
┌─────────────────┐
│data/processed/  │ Parquet
└────────┬────────┘
         │ load_historical.py (UPSERT)
         ▼
┌──────────────────────────────────┐
│       BigQuery (GCP)              │
│ ┌──────────────────────────────┐ │
│ │  prices_historical (raw)     │ │
│ └──────────┬───────────────────┘ │
│            │ feature_engineering_temporal.py
│            ▼
│ ┌──────────────────────────────┐ │
│ │ prices_ml_features (98 cols) │ │
│ └──────────┬───────────────────┘ │
└────────────┼─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│   Machine Learning Models        │
│  - Random Forest                 │
│  - Naive Bayes                   │
│  - Ensemble Methods              │
└──────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
crypto-etl-pipeline/
├── scripts/                    # ETL Pipeline
│   ├── extract.py             # Extracción de CoinGecko API
│   ├── transform.py           # Transformación y limpieza
│   ├── load_historical.py     # Carga a BigQuery (UPSERT)
│   ├── feature_engineering_temporal.py  # 98 features ML
│   ├── monitoring_dashboard.py          # Monitoreo de datos
│   └── analyze_data_status.py           # Análisis de cobertura
│
├── ml/                        # Machine Learning
│   ├── data_loader.py         # Carga datos desde BigQuery
│   ├── feature_engineer.py    # Feature selection y normalización
│   ├── model_trainer.py       # Entrenamiento de modelos
│   ├── model_evaluator.py     # Métricas y evaluación
│   ├── predictor.py           # Predicciones en tiempo real
│   ├── cross_validation.py    # Validación cruzada
│   ├── hyperparameter_tuning.py  # Grid Search
│   ├── backtesting.py         # Validación temporal
│   ├── walk_forward_validation.py  # Validación deslizante
│   ├── ensemble_predictor.py  # Modelos ensemble
│   └── optimize_model.py      # Script de optimización completo
│
├── test/                      # Tests (TDD estricto)
│   ├── test_*.py             # 17 tests de ML
│   └── ...                   # Tests de ETL y features
│
├── data/                     # Datos locales
│   ├── raw/                  # JSON raw de API
│   └── processed/            # Parquet procesados
│
├── models/                   # Modelos entrenados (.pkl)
│
├── .github/workflows/        # GitHub Actions
│   └── automated_etl_pipeline.yml  # Ejecución cada hora
│
├── venv/                     # Entorno virtual Python
│
├── Arquitecture.md           # ⚠️ REGLAS OBLIGATORIAS DEL PROYECTO
├── DOCUMENTACION.md          # 📚 Este archivo
├── PROGRESO_MODELO_ML.md     # 📊 Historial del modelo ML
└── requirements.txt          # Dependencias Python
```

---

## 🚀 Quick Start

### 1. Setup del Entorno

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Credenciales

**Variables de entorno requeridas (.env):**
```bash
# BigQuery (GCP)
PROJECT_ID=your-gcp-project
DATASET_NAME=crypto_data

# CoinGecko API (opcional, gratis por defecto)
COINGECKO_API_KEY=your-api-key  # Opcional
```

**Google Cloud Credentials:**
```bash
# Guardar service account JSON
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### 3. Ejecutar Pipeline ETL

```bash
# Extracción de datos
python scripts/extract.py

# Transformación
python scripts/transform.py

# Carga a BigQuery
python scripts/load_historical.py

# Feature engineering
python scripts/feature_engineering_temporal.py
```

### 4. Entrenar Modelo ML

```bash
# Optimización completa (Grid Search + Backtesting + Walk-Forward + Ensemble)
python ml/optimize_model.py

# O entrenar modelo específico
python ml/train_bitcoin_predictor.py  # Random Forest
python ml/train_bitcoin_naive_bayes.py  # Naive Bayes
```

### 5. Hacer Predicciones

```bash
# Predicción en tiempo real
python ml/predictor.py
```

### 6. Monitoreo

```bash
# Dashboard de cobertura y calidad de datos
python scripts/monitoring_dashboard.py
```

---

## 🔄 Automatización con GitHub Actions

El pipeline se ejecuta **automáticamente cada hora** via GitHub Actions.

**Workflow:** `.github/workflows/automated_etl_pipeline.yml`

**Ejecución:**
```
Schedule: "0 * * * *"  (cada hora)
├── Extract  (CoinGecko API)
├── Transform (limpieza)
├── Load (BigQuery UPSERT)
└── Feature Engineering (98 features)
```

**Monitoreo:**
- Ver logs en GitHub Actions → Workflows
- Check runs recientes en "Actions" tab

---

## 📊 Datasets en BigQuery

### Tabla: `prices_historical`

**Descripción:** Datos raw de precios de criptomonedas

**Esquema principal:**
- `id` (STRING): ID de la crypto (ej: 'bitcoin')
- `last_updated` (TIMESTAMP): Fecha/hora de snapshot
- `current_price` (FLOAT64): Precio actual en USD
- `market_cap`, `total_volume`, `high_24h`, `low_24h`, etc.
- `partition_date` (DATE): Partición por fecha

**Características:**
- **Particionado:** Por `partition_date` (optimiza queries)
- **Clustering:** Por `id` (agrupa cryptos similares)
- **UPSERT logic:** Evita duplicados basándose en (id, last_updated)

### Tabla: `prices_ml_features`

**Descripción:** Features temporales para Machine Learning (98 columnas)

**Features generadas:**
- **Lags:** price_lag_1h, price_lag_3h, price_lag_6h, price_lag_12h
- **Rolling stats:** price_ma_6h, price_ma_12h, price_ma_24h, price_std_24h
- **Momentum:** price_momentum_6h, rsi_14h, rsi_24h
- **Volatilidad:** volatility_24h, atr_14h, volatility_ratio
- **MACD:** macd, macd_signal, macd_histogram
- **Bollinger Bands:** bb_upper, bb_lower, bb_position
- **Volumen:** volume_ratio_6h_24h
- **Aceleración:** price_acceleration_6h, roc_12h
- **Temporales:** hour_sin, hour_cos, day_sin, day_cos

**Total:** 98 columnas

---

## 🧪 Testing (TDD Estricto)

**Filosofía:** Tests PRIMERO, código DESPUÉS (según `Arquitecture.md`)

### Ejecutar Tests

```bash
# Todos los tests
pytest test/ -v

# Tests específicos
pytest test/test_hyperparameter_tuning.py -v
pytest test/test_backtesting.py -v
pytest test/test_cross_validation.py -v
```

### Cobertura de Tests

- **ML Tests:** 17 tests (100% pasando)
- **ETL Tests:** Tests de feature engineering, data loader, etc.
- **Total:** 20+ tests

---

## 🛠️ Tecnologías y Dependencias

### Core
- **Python 3.13**
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **scikit-learn** - Machine Learning

### Cloud & Storage
- **google-cloud-bigquery** - GCP BigQuery
- **pyarrow** - Formato Parquet

### API & HTTP
- **requests** - Llamadas a CoinGecko API
- **python-dotenv** - Variables de entorno

### Testing
- **pytest** - Framework de testing
- **unittest** - Tests unitarios

### Instalación
```bash
pip install -r requirements.txt
```

---

## 📊 Monitoreo y Análisis

### Dashboard de Monitoreo

```bash
python scripts/monitoring_dashboard.py
```

**Información mostrada:**
- Total de registros acumulados
- Número de cryptos únicas
- Cobertura temporal (horas)
- Completeness score (% de datos válidos)
- Calidad de datos (nulls, inválidos)
- Top cryptos por cobertura
- Detección de gaps (>7 horas sin datos)

### Análisis de Features ML

```bash
python ml/feature_importance.py
```

**Salida:**
- Top 10 features más importantes
- Importancia relativa (%)
- Rankings y contribución al modelo

---

## ⚙️ Configuración Avanzada

### Cambiar Frecuencia de ETL

Editar `.github/workflows/automated_etl_pipeline.yml`:
```yaml
schedule:
  - cron: "0 * * * *"  # Cada hora (actual)
  - cron: "0 */2 * * *"  # Cada 2 horas
  - cron: "*/30 * * * *"  # Cada 30 minutos
```

### Agregar Nueva Crypto

Editar `scripts/extract.py`:
```python
CRYPTOS = [
    'bitcoin',
    'ethereum',
    'binancecoin',
    'nueva-crypto-id',  # Agregar aquí
]
```

### Cambiar Período de Datos ML

Editar `ml/optimize_model.py`:
```python
optimize_bitcoin_model(crypto_id='bitcoin', days=30)  # Cambiar days
```

---

## 🚨 Reglas de Desarrollo

**⚠️ IMPORTANTE: Lee `Arquitecture.md` antes de modificar código**

### Principios Obligatorios

1. **TDD Estricto:** Tests PRIMERO, código DESPUÉS
2. **Funciones pequeñas:** Máximo 50 líneas
3. **DRY:** No repetir código, extraer funciones
4. **Imports ordenados:** stdlib → third-party → local
5. **Logging:** Usar `logging`, NO `print()`
6. **Clean Code:** Nombres descriptivos, comentarios útiles

---

## 📈 Estado Actual del Sistema

### Datos Acumulados (24 Oct 2025)

- **Total registros:** 105,988
- **Cryptos únicas:** 294
- **Bitcoin registros:** 426
- **Cobertura temporal:** ~25 días (602 horas)
- **Completeness:** 100% (excelente)

### Modelo ML Actual

- **Mejor modelo:** Random Forest (Grid Search optimizado)
- **Accuracy:** 58.82% (test set)
- **Algoritmo:** RandomForestClassifier
- **Features usadas:** 21 features avanzadas
- **Última actualización:** 24 Oct 2025

Ver detalles completos en `PROGRESO_MODELO_ML.md`

---

## 🔧 Troubleshooting

### Error: "Unrecognized name: macd"

**Causa:** La tabla `prices_ml_features` no tiene las features avanzadas.

**Solución:**
```bash
python scripts/feature_engineering_temporal.py
```

### Error: BigQuery Credentials

**Causa:** Credenciales de GCP no configuradas.

**Solución:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### Error: Tests Fallan

**Causa:** Módulos no encontrados.

**Solución:** Agregar al inicio de los tests:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

---

## 📚 Recursos Adicionales

- **Arquitectura del proyecto:** `Arquitecture.md` (OBLIGATORIO leer)
- **Progreso ML:** `PROGRESO_MODELO_ML.md`
- **GitHub Actions:** `.github/workflows/automated_etl_pipeline.yml`

---

## 🤝 Contribución

**Antes de contribuir:**
1. Lee `Arquitecture.md` completo
2. Sigue TDD estricto (tests primero)
3. Ejecuta `pytest` antes de commit
4. Mantén funciones <50 líneas
5. Usa logging, NO prints

---

**Última actualización:** 2025-10-24  
**Mantenedor:** Sistema Crypto ETL Pipeline

