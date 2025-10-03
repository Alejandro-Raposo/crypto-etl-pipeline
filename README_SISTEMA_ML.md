# Sistema ML Robusto - Crypto ETL Pipeline

## Sistema Completo Implementado

### 1. Acumulacion Historica con UPSERT

**Script**: `scripts/load_historical.py`

Carga datos historicos a BigQuery evitando duplicados:
- **UPSERT automatico**: Combina datos existentes + nuevos, deduplica por clave compuesta
- **Clave compuesta**: `id + timestamp` (ej: `bitcoin_20251003120000`)
- **Particionamiento**: Por fecha para queries eficientes
- **Clustering**: Por id y partition_date
- **Compatible con tier gratuito**: No usa MERGE (DML), usa estrategia read-dedupe-write

**Ejecucion**:
```bash
python scripts/load_historical.py
```

**Tabla resultante**: `crypto_dataset.prices_historical`

### 2. Feature Engineering Temporal

**Script**: `scripts/feature_engineering_temporal.py`

Genera 86+ features temporales para ML:

**Lag Features**:
- Precios anteriores: 1h, 3h, 6h, 12h, 24h, 48h, 7d
- Volumenes anteriores: 1h, 24h

**Rolling Windows**:
- Medias moviles: 6h, 12h, 24h, 48h, 168h (7d)
- Desviacion estandar: Volatilidad en ventanas temporales
- Min/Max: Rangos de precio por ventana

**Indicadores Tecnicos**:
- RSI (Relative Strength Index): 14h, 24h
- Momentum: Cambios de precio en 6h, 12h, 24h
- Volatilidad: 24h, 48h, 168h

**Features Ciclicas**:
- hour_sin, hour_cos: Patron horario
- day_sin, day_cos: Patron semanal
- is_weekend: Indicador binario

**Features Relativas**:
- price_vs_ma: Precio vs media movil
- price_normalized: Normalizacion en ventana
- volume_change_pct: Cambio porcentual de volumen

**Ejecucion**:
```bash
python scripts/feature_engineering_temporal.py
```

**Tabla resultante**: `crypto_dataset.prices_ml_features`

### 3. Automatizacion con GitHub Actions

**Archivo**: `.github/workflows/automated_etl_pipeline.yml`

**Frecuencia**: Cada 6 horas (configurable)

**Pasos automaticos**:
1. Extract: API CoinGecko
2. Transform: Features basicas
3. Load Historical: UPSERT a BigQuery
4. Feature Engineering: Features temporales

**Configuracion**:
Ver `SETUP_AUTOMATED_PIPELINE.md` para detalles de configuracion

### 4. Tests de Integridad

**Archivo**: `test/test_data_integrity.py`

**Tests implementados**:
- Sin duplicados (composite_key unico)
- Timestamps validos (no futuros, no muy antiguos)
- Precios validos (> 0, no nulos)
- Multiples cryptos rastreadas
- Cobertura temporal calculada
- Datos recientes (< 7 dias)
- Deteccion de gaps en series temporales
- Calculo de completeness

**Ejecucion**:
```bash
python -m pytest test/test_data_integrity.py -v
```

### 5. Dashboard de Monitoreo

**Script**: `scripts/monitoring_dashboard.py`

**Metricas mostradas**:
- Estadisticas generales (registros, cryptos, fechas)
- Completeness Score (calidad de cobertura)
- Calidad de datos (nulls, invalidos)
- Top cryptos por cobertura
- Snapshots por fecha
- Deteccion de gaps

**Ejecucion**:
```bash
python scripts/monitoring_dashboard.py
```

## Flujo Completo del Sistema

```
┌─────────────────────────────────────────────────┐
│  GitHub Actions (cada 6 horas)                  │
│  - Ejecuta automaticamente                      │
│  - Notifica errores                             │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  1. EXTRACT (extract.py)                        │
│  - API CoinGecko                                │
│  - 250 cryptos                                  │
│  - Retry logic                                  │
│  → data/raw/coins_YYYYMMDD_HHMMSS.json         │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  2. TRANSFORM (transform.py)                    │
│  - Features basicas                             │
│  - Limpieza de datos                            │
│  → data/processed/coins_processed_*.parquet     │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  3. LOAD HISTORICAL (load_historical.py)        │
│  - UPSERT (evita duplicados)                    │
│  - Particionado por fecha                       │
│  - Clustered por id                             │
│  → BigQuery: prices_historical                  │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  4. FEATURE ENGINEERING (feature_eng_temp.py)   │
│  - 86+ features temporales                      │
│  - Lags, rolling windows, RSI                   │
│  → BigQuery: prices_ml_features                 │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  5. ML MODEL TRAINING (tu codigo)               │
│  - Time Series Forecasting                      │
│  - LSTM / Prophet / XGBoost                     │
│  → Predicciones de precio                       │
└─────────────────────────────────────────────────┘
```

## Acumulacion de Datos

### Proyeccion de Datos Historicos

Con ejecucion cada 6 horas:

| Periodo   | Snapshots | Registros Totales | Storage Estimado |
|-----------|-----------|-------------------|------------------|
| 1 dia     | 4         | 1,000             | ~200 KB          |
| 1 semana  | 28        | 7,000             | ~1.4 MB          |
| 1 mes     | 120       | 30,000            | ~6 MB            |
| 3 meses   | 360       | 90,000            | ~18 MB           |
| 6 meses   | 720       | 180,000           | ~36 MB           |
| 1 año     | 1,460     | 365,000           | ~73 MB           |

**Nota**: 250 cryptos por snapshot

### Datos Necesarios para ML Robusto

| Tarea ML                    | Minimo Recomendado | Optimo     |
|-----------------------------|--------------------|------------|
| Prediccion 1h               | 24h (4 snapshots)  | 7 dias     |
| Prediccion 24h              | 7 dias             | 30 dias    |
| Analisis de tendencias      | 30 dias            | 90 dias    |
| Deteccion de patrones       | 90 dias            | 180 dias   |
| Trading strategies          | 180 dias           | 1 año      |

## Comandos Utiles

### Ejecutar Pipeline Manual
```bash
python scripts/extract.py
python scripts/transform.py
python scripts/load_historical.py
python scripts/feature_engineering_temporal.py
```

### Ver Estado del Sistema
```bash
python scripts/monitoring_dashboard.py
```

### Ejecutar Tests
```bash
python -m pytest test/test_upsert_load.py -v
python -m pytest test/test_feature_engineering_temporal.py -v
python -m pytest test/test_data_integrity.py -v
python -m pytest test/ -v  # Todos los tests
```

### Verificar Tabla en BigQuery
```bash
python test/test_bigquery_table_status.py
```

## Estructura de Tablas en BigQuery

### prices_historical
- **250 columnas**: Todos los datos raw + calculados
- **Particionada**: Por partition_date (DATE)
- **Clustered**: Por id, partition_date
- **Sin duplicados**: Clave compuesta unica
- **Uso**: Almacenamiento historico completo

### prices_ml_features
- **86+ columnas**: Features temporales optimizadas para ML
- **Actualizacion**: Cada vez que se ejecuta feature engineering
- **Uso**: Entrenamiento directo de modelos ML

## Proximos Pasos

1. **Esperar acumulacion de datos** (minimo 7 dias)
2. **Entrenar modelo de prediccion**:
   - Time Series: ARIMA, Prophet
   - Deep Learning: LSTM, GRU
   - Ensemble: XGBoost + features temporales
3. **Implementar backtesting** con datos historicos
4. **Deploy modelo** con predicciones en tiempo real

## Notas Importantes

- **Tier gratuito BigQuery**: 100% compatible
- **API CoinGecko**: Respeta rate limits (1 call cada 6h)
- **Deduplicacion automatica**: No acumula datos duplicados
- **Gaps en datos**: Detectados automaticamente por monitoring
- **Tests obligatorios**: TDD implementado segun Arquitecture.md

