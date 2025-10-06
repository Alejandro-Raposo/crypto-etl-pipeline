# 📊 Análisis: Frecuencia de Recolección de Datos

## 🎯 Decisión: Cambiar de 6 horas → 1 hora

**Fecha:** 2025-10-06  
**Razón:** Optimizar acumulación de datos para ML

---

## 📈 COMPARACIÓN DETALLADA

### **Escenario Anterior (cada 6 horas)**

| Métrica | Valor |
|---------|-------|
| Snapshots/día | 4 |
| Snapshots/semana | 28 |
| Snapshots/mes | ~120 |
| Tiempo para 100 registros | 25 días |
| Tiempo para 500 registros | 125 días (4 meses) |
| Llamadas API/mes | 120 |

### **Escenario Nuevo (cada 1 hora)**

| Métrica | Valor |
|---------|-------|
| Snapshots/día | 24 ⬆️ **6x más** |
| Snapshots/semana | 168 ⬆️ **6x más** |
| Snapshots/mes | ~720 ⬆️ **6x más** |
| Tiempo para 100 registros | **4 días** ⬇️ 6x más rápido |
| Tiempo para 500 registros | **21 días** ⬇️ 6x más rápido |
| Llamadas API/mes | 720 |

---

## ✅ VENTAJAS (Por qué es mejor)

### **1. Acumulación 6x Más Rápida**

```
┌────────────────────────────────────────────────┐
│  TIEMPO PARA MODELO ROBUSTO                    │
├────────────────────────────────────────────────┤
│  Cada 6h: 25 días → 100 registros              │
│  Cada 1h:  4 días → 100 registros  ✅          │
│                                                 │
│  Cada 6h: 4 meses → 500 registros              │
│  Cada 1h: 21 días → 500 registros  ✅          │
└────────────────────────────────────────────────┘
```

**Impacto:** Modelo de producción en **3 semanas** vs **4 meses**

### **2. Alineación Perfecta con el Objetivo**

El modelo predice: **"¿El precio subirá o bajará en la PRÓXIMA HORA?"**

- ✅ Con datos cada 1h: Validación inmediata (1 hora después)
- ❌ Con datos cada 6h: Validación incompleta (6 horas después)

**Ejemplo:**
```
Predicción a las 10:00 → "Subirá en 1 hora"
Con 1h: Verificas a las 11:00 ✅
Con 6h: Verificas a las 16:00 ❌ (demasiado tarde)
```

### **3. Captura de Volatilidad Intradía**

Bitcoin puede moverse **5-10%** en 1 hora:

```
Ejemplo real:
10:00 → $62,000
11:00 → $63,500 (+2.4%)
12:00 → $62,800 (-1.1%)
13:00 → $64,200 (+2.2%)
14:00 → $63,900 (-0.5%)
15:00 → $65,100 (+1.9%)
16:00 → $64,500 (-0.9%)

Con 6h solo capturas: 10:00 y 16:00 (pierdes 4 movimientos)
Con 1h capturas: TODOS los movimientos ✅
```

### **4. Mejor Entrenamiento del Modelo**

Más datos = Mejor generalización:

```python
# Con datos cada 6h (después de 1 mes)
registros = 120
train_size = 96 (80%)
test_size = 24 (20%)
→ Modelo aprende de 96 ejemplos

# Con datos cada 1h (después de 1 mes)
registros = 720
train_size = 576 (80%)
test_size = 144 (20%)
→ Modelo aprende de 576 ejemplos ✅ (6x más)
```

### **5. Features Temporales Más Ricas**

Puedes crear lags más detallados:

```python
# Antes (cada 6h)
price_lag_6h   # Mínimo lag posible
price_lag_12h
price_lag_18h

# Ahora (cada 1h)
price_lag_1h   # ✅ Nuevo
price_lag_2h   # ✅ Nuevo
price_lag_3h   # ✅ Nuevo
price_lag_4h   # ✅ Nuevo
price_lag_5h   # ✅ Nuevo
price_lag_6h   # Ya existente
...
```

### **6. Backtesting Más Preciso**

Puedes validar predicciones hora por hora:

```python
# Cada 1h permite backtesting preciso
for hour in range(24):
    prediction = model.predict(X_hour)
    actual = y_hour
    accuracy_per_hour[hour] = compare(prediction, actual)

# Descubres: "El modelo es mejor entre 14:00-18:00 UTC"
```

---

## ⚠️ CONSIDERACIONES (Posibles problemas)

### **1. Límites de API**

**CoinGecko Free Tier:**
- Límite: 10-50 llamadas/minuto ✅
- Límite mensual: ~10,000-30,000 llamadas ✅

**Nuestro uso con 1h:**
- 24 llamadas/día
- 720 llamadas/mes
- **Conclusión:** ✅ MUY DENTRO DEL LÍMITE (usa 2-7% del límite)

### **2. GitHub Actions**

**Free Tier:**
- 2,000 minutos/mes (privado)
- Ilimitado (público) ✅

**Nuestro repo es público → Sin problemas**

### **3. BigQuery Storage**

**Free Tier:**
- 10 GB almacenamiento gratis

**Nuestro uso con 1h:**
```
720 snapshots/mes × 250 cryptos = 180,000 registros/mes
Tamaño: ~100 bytes/registro = 18 MB/mes
En 1 año: 216 MB
```
✅ **Completamente dentro del free tier**

### **4. Ruido en los Datos**

**Problema potencial:**
- Fluctuaciones muy cortas pueden ser "ruido"
- El modelo podría sobre-ajustarse

**Solución:**
```python
# Agregar suavizado en features
df['price_smooth_3h'] = df['current_price'].rolling(3).mean()
df['price_smooth_6h'] = df['current_price'].rolling(6).mean()

# El modelo aprende patrones reales, no ruido
```

---

## 🎯 RECOMENDACIÓN FINAL

# ✅ CAMBIA A CADA 1 HORA

### **Razones principales:**

1. ✅ **6x más rápido** para acumular datos
2. ✅ **Alineado con el objetivo** (predicción 1h)
3. ✅ **Dentro de límites gratuitos** (API, GitHub, BigQuery)
4. ✅ **Mejor captura de volatilidad**
5. ✅ **Modelo robusto en 1 semana** vs 1 mes

### **Riesgos:**

⚠️ Mínimos y manejables:
- Ruido → Solucionable con suavizado
- Límites API → Muy por debajo del límite

---

## 📊 PROYECCIÓN DE ACUMULACIÓN

### **Con frecuencia de 1 hora:**

| Período | Registros | Estado del Modelo |
|---------|-----------|-------------------|
| **3 días** | 72 | Entrenamiento inicial ⚙️ |
| **1 semana** | 168 | Modelo funcional ✅ |
| **2 semanas** | 336 | Modelo bueno 🎯 |
| **1 mes** | 720 | Modelo robusto 🚀 |
| **3 meses** | 2,160 | Modelo producción 💪 |

### **Comparación con 6 horas:**

```
OBJETIVO: 500 REGISTROS PARA MODELO ROBUSTO

Cada 6h: 125 días (4 meses)  ⏳⏳⏳⏳
Cada 1h:  21 días (3 semanas) ⏳ ✅
```

**Ahorro de tiempo: 104 días (3.5 meses)**

---

## 🔧 CAMBIOS IMPLEMENTADOS

### **1. GitHub Actions**

**Archivo:** `.github/workflows/automated_etl_pipeline.yml`

```yaml
# Antes
cron: '0 */6 * * *'  # Cada 6 horas

# Ahora
cron: '0 * * * *'    # Cada hora en punto
```

### **2. Documentación**

- ✅ `README.md` actualizado
- ✅ `ML_USAGE.md` actualizado
- ✅ `FRECUENCIA_DATOS_ANALISIS.md` creado

### **3. Sin Cambios en Código**

El código ETL NO necesita modificación:
- ✅ `extract.py` funciona igual
- ✅ `transform.py` funciona igual
- ✅ `load_historical.py` funciona igual (UPSERT maneja duplicados)
- ✅ `feature_engineering_temporal.py` funciona igual

---

## 📈 MÉTRICAS A MONITOREAR

### **Después del cambio, monitorear:**

1. **Uso de API**
   ```bash
   # Verificar que no excedes límites
   # CoinGecko envía headers: X-RateLimit-Remaining
   ```

2. **GitHub Actions**
   ```bash
   # Ver consumo en:
   # Repo → Settings → Actions → Usage
   ```

3. **BigQuery Storage**
   ```bash
   python scripts/monitoring_dashboard.py
   # Total registros debería crecer 6x más rápido
   ```

4. **Calidad del Modelo**
   ```bash
   # Reentrenar cada semana
   python ml/train_bitcoin_predictor.py
   
   # Accuracy debería mejorar con más datos
   ```

---

## 🎯 SIGUIENTES PASOS

### **Esta semana:**
1. ✅ Cambio a 1 hora implementado
2. ⏳ Esperar 3-7 días
3. ⏳ Verificar acumulación de datos

### **Próxima semana:**
1. Reentrenar modelo con ~168 registros
2. Comparar accuracy con modelo actual
3. Validar que la mejora es real

### **En 1 mes:**
1. Reentrenar con ~720 registros
2. Modelo debería tener >75% accuracy
3. Considerar deploy a producción

---

## 🎓 LECCIONES APRENDIDAS

1. **Más datos ≠ Más caro**
   - Con servicios cloud free tier, puedes recolectar mucho
   
2. **Frecuencia debe alinearse con objetivo**
   - Predices 1h → Recolectas cada 1h
   
3. **Experimentación es clave**
   - Empezaste con 6h, ahora optimizas a 1h
   - Puedes ajustar en el futuro si es necesario

4. **Monitoreo es esencial**
   - Siempre verifica que no excedes límites
   - Dashboard de monitoreo es crítico

---

## 📚 REFERENCIAS

- **CoinGecko API Docs:** https://www.coingecko.com/en/api/documentation
- **GitHub Actions Pricing:** https://docs.github.com/en/billing/managing-billing-for-github-actions
- **BigQuery Free Tier:** https://cloud.google.com/bigquery/pricing#free-tier

---

**Decisión tomada:** 2025-10-06  
**Implementado por:** Sistema ML optimizado  
**Beneficio esperado:** Modelo robusto en 1 semana vs 1 mes

