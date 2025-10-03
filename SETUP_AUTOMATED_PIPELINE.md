# Configuracion del Pipeline Automatizado

## GitHub Actions - Ejecucion Cada 6 Horas

### 1. Configurar Secret en GitHub

1. Ve a tu repositorio en GitHub
2. Settings > Secrets and variables > Actions
3. Click en "New repository secret"
4. Nombre: `GCP_SERVICE_ACCOUNT_KEY`
5. Valor: Copia el contenido completo del archivo JSON de tu service account de GCP

### 2. Verificar Configuracion

El workflow esta configurado para:
- **Ejecutarse automaticamente** cada 6 horas (`cron: '0 */6 * * *'`)
- **Ejecutarse manualmente** desde GitHub Actions UI
- **Pasos del pipeline**:
  1. Extract: Obtiene datos de CoinGecko API
  2. Transform: Procesa y calcula features basicas
  3. Load Historical: Carga a BigQuery con UPSERT (evita duplicados)
  4. Feature Engineering: Genera features temporales (lags, rolling windows, RSI, etc.)

### 3. Modificar Frecuencia

Edita el archivo `.github/workflows/automated_etl_pipeline.yml`:

```yaml
schedule:
  - cron: '0 */1 * * *'  # Cada 1 hora
  - cron: '0 */4 * * *'  # Cada 4 horas
  - cron: '0 0 * * *'    # Diario a medianoche
  - cron: '0 0,12 * * *' # 2 veces al dia (00:00 y 12:00)
```

### 4. Ejecucion Manual

1. Ve a Actions en tu repositorio
2. Selecciona "Automated Crypto ETL Pipeline"
3. Click en "Run workflow"
4. Selecciona branch y ejecuta

### 5. Monitoreo

- Revisa los logs en GitHub Actions > Workflow runs
- Descarga artifacts (datos raw/procesados) desde cada ejecucion
- Verifica tablas en BigQuery:
  - `prices_historical`: Todos los snapshots historicos
  - `prices_ml_features`: Features temporales listas para ML

### 6. Estructura de Tablas

#### prices_historical
- **Particionada por fecha** (partition_date)
- **Clustered por** id, partition_date
- **UPSERT automatico**: No duplica datos con misma clave
- **Clave compuesta**: `id + timestamp`

#### prices_ml_features
- **86+ features temporales**:
  - Lags: 1h, 3h, 6h, 12h, 24h, 48h, 7d
  - Moving Averages: 6h, 12h, 24h, 48h, 168h
  - Volatilidad: 24h, 48h, 168h
  - RSI: 14h, 24h
  - Momentum: 6h, 12h, 24h
  - Features ciclicas: hour_sin, hour_cos, day_sin, day_cos

### 7. Estimacion de Datos

Con ejecucion cada 6 horas:
- **Dia 1**: 4 snapshots (6am, 12pm, 6pm, 12am)
- **Semana 1**: 28 snapshots
- **Mes 1**: 120 snapshots
- **Datos por snapshot**: 250 cryptos

**Total despues de 1 mes**: ~30,000 registros historicos

### 8. Costos BigQuery (Tier Gratuito)

- Storage: 10 GB gratis/mes
- Queries: 1 TB procesado gratis/mes
- **Estimado pipeline**: < 100 MB/mes storage, < 10 GB/mes queries
- **Resultado**: 100% dentro del tier gratuito

### 9. Backup Local

Los artifacts se guardan 7 dias en GitHub:
- `data/raw/*.json`
- `data/processed/*.parquet`

Para backup permanente, descarga periodicamente desde Actions > Artifacts

### 10. Troubleshooting

**Error: Billing not enabled**
- Verifica que el proyecto GCP tenga billing habilitado
- O usa estrategia WRITE_TRUNCATE en lugar de MERGE

**Error: Credentials not found**
- Verifica que el secret GCP_SERVICE_ACCOUNT_KEY este configurado
- Confirma que el JSON sea valido

**Error: API rate limit**
- CoinGecko API gratuita: 10-50 calls/min
- El pipeline hace 1 call cada 6 horas (seguro)

**Datos insuficientes para ML**
- Espera minimo 24 horas de datos (4 snapshots)
- Optimo: 7+ dias de datos para features robustos

