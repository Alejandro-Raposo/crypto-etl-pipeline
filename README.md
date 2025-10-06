# 🚀 Crypto ETL Pipeline + ML Predictor

Pipeline ETL automatizado que extrae datos de criptomonedas desde CoinGecko API, los procesa, almacena en BigQuery, genera features temporales para Machine Learning y entrena modelos para predecir movimientos de precios.

---

## 📖 Descripción

Sistema completo de **ETL + Machine Learning** para análisis y predicción de precios de criptomonedas:

- **ETL Pipeline:** Extrae, transforma y carga datos cada 6 horas (GitHub Actions)
- **Acumulación Histórica:** Almacena datos con UPSERT logic para evitar duplicados
- **Feature Engineering Temporal:** Genera 86+ features (lags, rolling stats, RSI, momentum)
- **ML Predictor:** Modelo Random Forest que predice dirección de precio (UP/DOWN)
- **Test-Driven Development:** 62 tests automatizados (100% pasando)

---

## 🎯 Objetivos

1. ✅ **Automatizar** la extracción de datos de criptomonedas
2. ✅ **Acumular historial** para análisis temporal robusto
3. ✅ **Generar features ML** avanzadas (lags, rolling windows, RSI)
4. ✅ **Entrenar modelos** de predicción de precios
5. ✅ **Hacer predicciones** en tiempo real
6. ✅ **Monitorear** calidad y cobertura de datos

---

## 🏗️ Arquitectura

```
┌─────────────────┐
│  CoinGecko API  │
└────────┬────────┘
         │ extract.py (cada 6h)
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
│ │ prices_historical            │ │ ← Datos raw acumulados
│ │ (partitioned by date)        │ │
│ └──────────────────────────────┘ │
│              │                    │
│              │ feature_engineering_temporal.py
│              ▼                    │
│ ┌──────────────────────────────┐ │
│ │ prices_ml_features           │ │ ← 86+ features ML
│ │ (lag, rolling, RSI, momentum)│ │
│ └──────────────────────────────┘ │
└──────────────────────────────────┘
         │
         │ ml/train_bitcoin_predictor.py
         ▼
┌─────────────────┐
│  models/*.pkl   │ ← Modelos entrenados
└────────┬────────┘
         │ ml/predictor.py
         ▼
┌─────────────────┐
│  PREDICCIÓN     │ UP/DOWN + Confianza
└─────────────────┘
```

---

## 📂 Estructura del Proyecto

```
    crypto-etl-pipeline/
├── .github/workflows/
│   └── automated_etl_pipeline.yml    # Ejecuta ETL cada 6 horas
├── data/
│   ├── raw/                          # JSON de CoinGecko (gitignored)
│   └── processed/                    # Parquet transformados (gitignored)
├── ml/
│   ├── data_loader.py                # Carga datos de BigQuery
│   ├── feature_engineer.py           # Prepara features ML
│   ├── model_trainer.py              # Entrena Random Forest
│   ├── model_evaluator.py            # Evalúa métricas
│   ├── predictor.py                  # Hace predicciones
│   └── train_bitcoin_predictor.py    # Script de entrenamiento
├── models/
│   └── bitcoin_predictor_*.pkl       # Modelos entrenados (gitignored)
├── scripts/
│   ├── extract.py                    # Extrae de CoinGecko API
│   ├── transform.py                  # Transforma con Pandas
│   ├── load_historical.py            # Carga a BigQuery (UPSERT)
│   ├── feature_engineering_temporal.py  # Genera features ML
│   └── monitoring_dashboard.py       # Monitoreo de datos
├── test/
│   ├── test_ml_*.py                  # 27 tests ML
│   ├── test_upsert_load.py           # 10 tests UPSERT
│   ├── test_feature_engineering_temporal.py  # 13 tests
│   └── test_data_integrity.py        # 12 tests integridad
├── Arquitecture.md                   # Reglas de desarrollo (TDD)
├── ML_USAGE.md                       # Documentación ML completa
├── README.md                         # Este archivo
└── requirements.txt                  # Dependencias Python
```

---

## 🚀 Inicio Rápido

### 1. Clonar repositorio

```bash
git clone https://github.com/Alejandro-Raposo/crypto-etl-pipeline.git
cd crypto-etl-pipeline
```

### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales

Crear archivo `.env` en la raíz:

```env
ETL_SERVICE_ACOUNT_CREDENTIALS_ROUTE=path/to/gcp_credentials.json
```

### 5. Ejecutar pipeline ETL (manual)

```bash
# Extraer
python scripts/extract.py

# Transformar
python scripts/transform.py

# Cargar
python scripts/load_historical.py

# Features ML
python scripts/feature_engineering_temporal.py
```

### 6. Entrenar modelo ML

```bash
python ml/train_bitcoin_predictor.py
```

### 7. Hacer predicción

```bash
python ml/predictor.py
```

**Salida:**
```
============================================================
PREDICCIÓN COMPLETADA
============================================================

Dirección: UP ↗️
Confianza: 57.78%

Predicción: El precio SUBE (Confianza: 57.8%)
============================================================
```

---

## 🤖 Sistema de Machine Learning

### Objetivo

Predecir si el precio de **Bitcoin subirá o bajará** en la próxima hora.

### Modelo

- **Tipo:** Clasificación binaria (UP=1, DOWN=0)
- **Algoritmo:** Random Forest (100 árboles)
- **Features:** 12 (lags, medias móviles, RSI, volatilidad, momentum)
- **Normalización:** MinMaxScaler (0-1)

### Métricas Actuales

```
Accuracy:  66.67%
Precision: 66.67%
Recall:    100.00%
F1-Score:  80.00%
```

⚠️ **Nota:** Modelo entrenado con 13 registros. Mejorará con más datos históricos.

### Documentación Completa

Ver **[ML_USAGE.md](ML_USAGE.md)** para:
- Uso programático (Python)
- Interpretación de resultados
- Workflow recomendado
- 27 tests implementados

---

## ⚙️ Automatización (GitHub Actions)

El pipeline se ejecuta **automáticamente cada 6 horas**:

1. ✅ Extrae datos de CoinGecko API
2. ✅ Transforma y limpia
3. ✅ Carga a BigQuery con UPSERT
4. ✅ Genera features ML temporales

### Configurar GitHub Actions

1. Crear secret `GCP_SERVICE_ACCOUNT_KEY` en GitHub:
   - Settings → Secrets and variables → Actions → New secret
   - Pegar el contenido completo del JSON de credenciales GCP

2. El workflow se ejecuta automáticamente (cron: `0 */6 * * *`)

3. También se puede ejecutar manualmente:
   - Actions → Automated ETL Pipeline → Run workflow

---

## 📊 Monitoreo

### Ver cobertura de datos

```bash
python scripts/monitoring_dashboard.py
```

**Salida:**
```
========================================
DASHBOARD: COBERTURA TEMPORAL DE DATOS
========================================

Total registros: 13
Cryptos únicas: 1 (Bitcoin)
Fechas únicas: 13
Horas cubiertas: ~13 horas

Registro más antiguo: 2025-10-01 12:00:00
Registro más reciente: 2025-10-06 18:00:00

Completeness Score: 85.0%
```

---

## 🧪 Tests (62 tests - 100% pasando)

### Ejecutar todos los tests

```bash
pytest test/ -v
```

### Tests por módulo

```bash
# Tests ML (27 tests)
pytest test/test_ml_*.py -v

# Tests ETL (35 tests)
pytest test/test_upsert_load.py -v
pytest test/test_feature_engineering_temporal.py -v
pytest test/test_data_integrity.py -v
```

### Cobertura

- ✅ **ETL:** UPSERT logic, feature engineering, integridad temporal
- ✅ **ML:** Data loader, feature engineer, trainer, evaluator, predictor
- ✅ **BigQuery:** Queries, schema, partitioning

---

## 📈 Datasets en BigQuery

### `crypto_dataset.prices_historical`

Tabla particionada por fecha con datos raw acumulados:

- `id`, `symbol`, `name`
- `current_price`, `market_cap`, `volume_24h`
- `last_updated`, `partition_date`
- `composite_key` (deduplicación)

### `crypto_dataset.prices_ml_features`

Tabla con 86+ features para ML:

- Lags: 1h, 3h, 6h, 12h, 24h, 48h, 7d
- Rolling stats: mean, std, min, max (6h, 12h, 24h, 48h, 168h)
- RSI: 14h, 24h
- Momentum, volatilidad, cambios porcentuales
- Features cíclicas: hora, día de la semana (sin/cos)

---

## 🔧 Tecnologías Utilizadas

| Categoría | Tecnología |
|-----------|-----------|
| **Lenguaje** | Python 3.13 |
| **ETL** | Pandas, Requests, Pyarrow |
| **Cloud** | Google Cloud BigQuery |
| **ML** | scikit-learn (Random Forest) |
| **Automatización** | GitHub Actions |
| **Testing** | pytest |
| **Gestión deps** | pip, venv |

---

## 📋 Dependencias (requirements.txt)

```
requests>=2.32.0
pandas>=2.3.0
pyarrow>=21.0.0
google-cloud-bigquery>=3.38.0
pandas-gbq>=0.29.0
db-dtypes>=1.4.0
sqlalchemy>=2.0.0
python-dateutil>=2.9.0
python-dotenv>=1.1.0
numpy>=2.3.0
scikit-learn>=1.5.0
```

---

## 📖 Reglas de Desarrollo

Ver **[Arquitecture.md](Arquitecture.md)** para:

- ✅ **TDD estricto** (tests ANTES del código)
- ✅ Código limpio y organizado
- ✅ Rendimiento óptimo
- ✅ Sin código duplicado (DRY)
- ✅ Funciones cortas (<50 líneas)

**Regla #7 (TDD)** es la más crítica: los tests se diseñan ANTES de implementar funcionalidad.

---

## 🎯 Roadmap

### Completado ✅

- [x] ETL Pipeline automatizado
- [x] Acumulación histórica con UPSERT
- [x] Feature Engineering temporal (86+ features)
- [x] Modelo ML Random Forest
- [x] Sistema de predicción en tiempo real
- [x] 62 tests automatizados
- [x] Monitoreo de cobertura de datos
- [x] GitHub Actions (cada 6 horas)

### Próximos Pasos 🔲

- [ ] Esperar acumulación de datos (2-3 semanas)
- [ ] Reentrenar modelo con más snapshots
- [ ] Probar con otras cryptos (Ethereum, Solana)
- [ ] Implementar cross-validation
- [ ] Dashboard web interactivo (Streamlit/Dash)
- [ ] API REST para predicciones
- [ ] Probar LSTM para secuencias temporales
- [ ] Ensemble de modelos

---

## ⚠️ Advertencias

Este sistema es **educativo/experimental**:

- ⚠️ NO usar para decisiones financieras reales
- ⚠️ El mercado cripto es altamente volátil
- ⚠️ Resultados pasados NO garantizan resultados futuros
- ⚠️ Modelo actual tiene datos limitados (13 registros)

---

## 👨‍💻 Autor

**Alejandro Raposo**

- GitHub: [@Alejandro-Raposo](https://github.com/Alejandro-Raposo)
- Repositorio: [crypto-etl-pipeline](https://github.com/Alejandro-Raposo/crypto-etl-pipeline)

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo licencia MIT.

---

## 🎓 Aprendizajes Clave

Este proyecto demuestra:

1. ✅ **ETL end-to-end** con automatización
2. ✅ **Cloud integration** (GCP BigQuery)
3. ✅ **Machine Learning** aplicado a finanzas
4. ✅ **Test-Driven Development** estricto
5. ✅ **CI/CD** con GitHub Actions
6. ✅ **Código profesional** y escalable
7. ✅ **Monitoreo** y validación de datos

---

**Sistema ETL + ML completo y profesional** 🚀✨

Para más información sobre el sistema ML, ver **[ML_USAGE.md](ML_USAGE.md)**
