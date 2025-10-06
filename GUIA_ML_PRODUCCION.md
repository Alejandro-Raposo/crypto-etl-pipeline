# 🎓 GUÍA MAESTRA: ML en Producción para Trading de Criptomonedas

**Objetivo:** Transformar tu modelo de "funcional" a "producción" con las mejores prácticas de ML aplicado a finanzas.

---

## 📚 ÍNDICE

1. [Feature Engineering Avanzado](#1-feature-engineering-avanzado)
2. [Optimización del Modelo](#2-optimización-del-modelo)
3. [Reentrenamiento Continuo y Drift Detection](#3-reentrenamiento-continuo-y-drift-detection)
4. [Evaluación sin Data Leakage](#4-evaluación-sin-data-leakage)
5. [Estrategia Multi-Modelo vs Modelo Único](#5-estrategia-multi-modelo-vs-modelo-único)

---

# 1. FEATURE ENGINEERING AVANZADO

## 🎯 Objetivo

Crear features que capturen patrones predictivos reales, no ruido.

---

## 📊 1.1 Categorías de Features

### **A. Features Temporales (Lags)**

**Principio:** Los precios pasados influyen en precios futuros.

```python
# ml/advanced_features.py

def create_lag_features(df, col='current_price', lags=[1, 2, 3, 6, 12, 24, 48, 168]):
    """
    Crea features de lag (valores pasados).
    
    Lags recomendados para crypto (datos cada 1h):
    - 1h, 2h, 3h: Tendencia muy corta
    - 6h, 12h: Tendencia corta
    - 24h, 48h: Tendencia media
    - 168h (1 semana): Tendencia larga
    """
    for lag in lags:
        df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    return df


def create_price_differences(df, col='current_price', periods=[1, 3, 6, 24]):
    """
    Diferencias de precio (cambio absoluto).
    Útil para detectar momentum.
    """
    for period in periods:
        df[f'{col}_diff_{period}h'] = df[col] - df[col].shift(period)
    
    return df


def create_price_changes_pct(df, col='current_price', periods=[1, 3, 6, 12, 24]):
    """
    Cambios porcentuales de precio.
    Mejor que diferencias absolutas para comparar cryptos.
    """
    for period in periods:
        df[f'{col}_pct_change_{period}h'] = (
            (df[col] - df[col].shift(period)) / df[col].shift(period) * 100
        )
    
    return df
```

**Test (TDD):**

```python
# test/test_advanced_features.py

def test_lag_features_correctas():
    """Verifica que los lags se crean correctamente"""
    df = pd.DataFrame({
        'current_price': [100, 102, 104, 106, 108],
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='1H')
    })
    
    result = create_lag_features(df, lags=[1, 2])
    
    # Lag 1h debe ser el valor anterior
    assert result.iloc[1]['current_price_lag_1h'] == 100
    assert result.iloc[2]['current_price_lag_1h'] == 102
    
    # Lag 2h debe ser el valor 2 períodos atrás
    assert result.iloc[2]['current_price_lag_2h'] == 100
```

---

### **B. Features de Rolling Window (Ventanas Móviles)**

**Principio:** Estadísticas sobre ventanas de tiempo revelan tendencias.

```python
def create_rolling_mean(df, col='current_price', windows=[3, 6, 12, 24, 48, 168]):
    """
    Medias móviles.
    Detectan la tendencia general.
    """
    for window in windows:
        df[f'{col}_ma_{window}h'] = df[col].rolling(window=window).mean()
    
    return df


def create_rolling_std(df, col='current_price', windows=[6, 12, 24, 48]):
    """
    Desviación estándar móvil.
    Mide la volatilidad.
    """
    for window in windows:
        df[f'{col}_std_{window}h'] = df[col].rolling(window=window).std()
    
    return df


def create_rolling_min_max(df, col='current_price', windows=[24, 48, 168]):
    """
    Mínimos y máximos en ventanas.
    Detectan soportes y resistencias.
    """
    for window in windows:
        df[f'{col}_min_{window}h'] = df[col].rolling(window=window).min()
        df[f'{col}_max_{window}h'] = df[col].rolling(window=window).max()
    
    return df


def create_bollinger_bands(df, col='current_price', window=24, num_std=2):
    """
    Bandas de Bollinger.
    Detectan sobrecompra/sobreventa.
    """
    ma = df[col].rolling(window=window).mean()
    std = df[col].rolling(window=window).std()
    
    df[f'{col}_bb_upper'] = ma + (std * num_std)
    df[f'{col}_bb_lower'] = ma - (std * num_std)
    df[f'{col}_bb_position'] = (df[col] - df[f'{col}_bb_lower']) / (
        df[f'{col}_bb_upper'] - df[f'{col}_bb_lower']
    )
    
    return df
```

---

### **C. Indicadores Técnicos**

**Principio:** Indicadores usados por traders tienen poder predictivo.

```python
def create_rsi(df, col='current_price', periods=[14, 24]):
    """
    Relative Strength Index.
    Detecta sobrecompra (>70) y sobreventa (<30).
    """
    for period in periods:
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}h'] = 100 - (100 / (1 + rs))
    
    return df


def create_macd(df, col='current_price', fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence.
    Detecta cambios de momentum.
    """
    ema_fast = df[col].ewm(span=fast).mean()
    ema_slow = df[col].ewm(span=slow).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def create_momentum_indicators(df, col='current_price', periods=[6, 12, 24]):
    """
    Indicadores de momentum.
    """
    for period in periods:
        # Rate of Change
        df[f'roc_{period}h'] = (
            (df[col] - df[col].shift(period)) / df[col].shift(period) * 100
        )
        
        # Momentum simple
        df[f'momentum_{period}h'] = df[col] - df[col].shift(period)
    
    return df


def create_volume_indicators(df, price_col='current_price', vol_col='volume_24h'):
    """
    Indicadores basados en volumen.
    El volumen confirma movimientos de precio.
    """
    # Volumen relativo
    df['volume_ratio_24h'] = df[vol_col] / df[vol_col].rolling(24).mean()
    
    # On-Balance Volume (simplificado)
    df['obv'] = (np.sign(df[price_col].diff()) * df[vol_col]).cumsum()
    
    # Volume-Weighted Average Price
    df['vwap_24h'] = (
        (df[price_col] * df[vol_col]).rolling(24).sum() / 
        df[vol_col].rolling(24).sum()
    )
    
    return df
```

---

### **D. Features de Volatilidad**

**Principio:** La volatilidad predice movimientos futuros.

```python
def create_volatility_features(df, col='current_price', windows=[6, 12, 24, 48]):
    """
    Múltiples medidas de volatilidad.
    """
    for window in windows:
        # Desviación estándar (volatilidad clásica)
        df[f'volatility_std_{window}h'] = (
            df[col].pct_change().rolling(window).std() * 100
        )
        
        # Rango (max - min)
        df[f'price_range_{window}h'] = (
            df[col].rolling(window).max() - df[col].rolling(window).min()
        )
        
        # Average True Range (ATR)
        high = df[col].rolling(window).max()
        low = df[col].rolling(window).min()
        close_prev = df[col].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{window}h'] = tr.rolling(window).mean()
    
    return df


def create_realized_volatility(df, col='current_price', window=24):
    """
    Volatilidad realizada (usada en finanzas cuantitativas).
    """
    returns = np.log(df[col] / df[col].shift(1))
    df[f'realized_vol_{window}h'] = returns.rolling(window).std() * np.sqrt(window)
    
    return df
```

---

### **E. Features de Interacción**

**Principio:** Combinaciones de features revelan patrones ocultos.

```python
def create_interaction_features(df):
    """
    Features que combinan información de múltiples fuentes.
    """
    # Relación precio vs media móvil
    df['price_vs_ma24'] = df['current_price'] / df['current_price_ma_24h']
    df['price_vs_ma168'] = df['current_price'] / df['current_price_ma_168h']
    
    # Momentum vs volatilidad
    df['momentum_vol_ratio'] = (
        df['current_price_pct_change_6h'] / df['volatility_std_24h']
    )
    
    # RSI diferencial
    df['rsi_diff'] = df['rsi_14h'] - df['rsi_24h']
    
    # Velocidad de cambio del momentum
    df['momentum_acceleration'] = (
        df['momentum_6h'] - df['momentum_6h'].shift(6)
    )
    
    # Posición en el rango de precios
    df['price_position_24h'] = (
        (df['current_price'] - df['current_price_min_24h']) /
        (df['current_price_max_24h'] - df['current_price_min_24h'])
    )
    
    return df
```

---

### **F. Features Temporales Cíclicas**

**Principio:** El tiempo del día/semana tiene patrones.

```python
def create_temporal_features(df, timestamp_col='last_updated'):
    """
    Features basadas en el momento del día/semana.
    Los mercados tienen patrones horarios.
    """
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    
    # Transformación cíclica (mejor que valores discretos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Features binarias para horarios clave
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_asian_hours'] = df['hour'].between(0, 8).astype(int)
    df['is_european_hours'] = df['hour'].between(8, 16).astype(int)
    df['is_us_hours'] = df['hour'].between(14, 22).astype(int)
    
    return df
```

---

## ⚠️ 1.2 Advertencias Críticas

### **A. Evitar Data Leakage**

```python
# ❌ INCORRECTO (usa datos futuros)
df['future_price'] = df['current_price'].shift(-1)  # ¡Esto es trampa!
df['target'] = (df['future_price'] > df['current_price']).astype(int)

# ✅ CORRECTO (solo usa datos pasados)
def create_target_safe(df):
    """
    Target creado DESPUÉS de features.
    Features solo usan datos pasados (shift positivo).
    """
    df_features = df.copy()
    
    # Crear todas las features (solo datos pasados)
    df_features = create_lag_features(df_features)
    df_features = create_rolling_mean(df_features)
    # ... más features
    
    # Crear target (usa futuro, pero está separado)
    df_features['price_future_1h'] = df_features['current_price'].shift(-1)
    df_features['target'] = (
        df_features['price_future_1h'] > df_features['current_price']
    ).astype(int)
    
    # Eliminar última fila (no tiene futuro conocido)
    df_features = df_features[:-1]
    
    return df_features
```

### **B. Manejo de NaNs**

```python
def handle_nans_correctly(df):
    """
    Manejo profesional de NaNs.
    """
    # 1. Identificar columnas con NaNs
    nan_counts = df.isnull().sum()
    print(f"Columnas con NaNs: {nan_counts[nan_counts > 0]}")
    
    # 2. Estrategia por tipo de feature
    
    # Lags: forward fill (usar último valor conocido)
    lag_cols = [col for col in df.columns if 'lag_' in col]
    df[lag_cols] = df[lag_cols].fillna(method='ffill')
    
    # Rolling stats: usar ventana mínima o eliminar filas iniciales
    rolling_cols = [col for col in df.columns if any(x in col for x in ['ma_', 'std_', 'rsi_'])]
    # Opción 1: Eliminar filas iniciales
    max_window = 168  # Tu ventana más grande
    df = df.iloc[max_window:]
    
    # Opción 2: Rellenar con media (menos recomendado)
    # df[rolling_cols] = df[rolling_cols].fillna(df[rolling_cols].mean())
    
    # 3. Verificar que no queden NaNs
    assert df.isnull().sum().sum() == 0, "Todavía hay NaNs!"
    
    return df
```

---

## 📋 1.3 Checklist de Features

```python
# ml/feature_checklist.py

FEATURE_CATEGORIES = {
    'lags': [
        'price_lag_1h', 'price_lag_3h', 'price_lag_6h', 
        'price_lag_12h', 'price_lag_24h', 'price_lag_48h'
    ],
    'rolling_mean': [
        'price_ma_6h', 'price_ma_12h', 'price_ma_24h', 
        'price_ma_48h', 'price_ma_168h'
    ],
    'rolling_std': [
        'price_std_6h', 'price_std_12h', 'price_std_24h'
    ],
    'momentum': [
        'roc_6h', 'roc_12h', 'roc_24h',
        'momentum_6h', 'momentum_12h'
    ],
    'technical': [
        'rsi_14h', 'rsi_24h', 
        'macd', 'macd_signal', 'macd_hist'
    ],
    'volatility': [
        'volatility_std_24h', 'atr_24h', 'realized_vol_24h'
    ],
    'interaction': [
        'price_vs_ma24', 'momentum_vol_ratio', 'rsi_diff'
    ],
    'temporal': [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_asian_hours', 'is_us_hours'
    ]
}

def validate_features(df):
    """Verifica que todas las features críticas existan"""
    missing = []
    for category, features in FEATURE_CATEGORIES.items():
        for feature in features:
            if feature not in df.columns:
                missing.append(f"{category}: {feature}")
    
    if missing:
        raise ValueError(f"Features faltantes: {missing}")
    
    return True
```

---

# 2. OPTIMIZACIÓN DEL MODELO

## 🎯 Objetivo

Encontrar la configuración óptima del modelo para maximizar precisión predictiva.

---

## 📊 2.1 Validación Temporal (NO Cross-Validation Estándar)

**⚠️ CRÍTICO:** En series temporales, NO puedes usar K-Fold estándar.

### **Por qué K-Fold es INCORRECTO:**

```python
# ❌ INCORRECTO para series temporales
from sklearn.model_selection import cross_val_score

# Esto mezcla pasado con futuro → data leakage
scores = cross_val_score(model, X, y, cv=5)  # ¡ERROR!
```

### **✅ CORRECTO: Time Series Split**

```python
# ml/temporal_validation.py

from sklearn.model_selection import TimeSeriesSplit

def time_series_validation(model, X, y, n_splits=5):
    """
    Validación respetando el orden temporal.
    
    Ejemplo con 5 splits:
    Train: [----]
    Test:        [--]
    
    Train: [-------]
    Test:           [--]
    
    Train: [----------]
    Test:              [--]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'all_scores': scores
    }
```

### **✅ MEJOR: Walk-Forward Validation**

```python
def walk_forward_validation(model, X, y, train_size=100, test_size=24):
    """
    Validación walk-forward (más realista).
    
    Simula entrenamiento continuo:
    1. Entrena con datos [0:100]
    2. Predice [100:124]
    3. Reentrena con [0:124]
    4. Predice [124:148]
    ...
    """
    scores = []
    predictions = []
    
    for i in range(train_size, len(X) - test_size, test_size):
        # Ventana de entrenamiento
        X_train = X.iloc[i-train_size:i]
        y_train = y.iloc[i-train_size:i]
        
        # Ventana de test
        X_test = X.iloc[i:i+test_size]
        y_test = y.iloc[i:i+test_size]
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Evaluar
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        predictions.extend(y_pred)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'all_scores': scores,
        'predictions': predictions
    }
```

---

## 🔧 2.2 Grid Search con Validación Temporal

```python
# ml/hyperparameter_tuning.py

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import itertools

def temporal_grid_search(X, y, param_grid, n_splits=5):
    """
    Grid search respetando orden temporal.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Generar todas las combinaciones
    param_combinations = [
        dict(zip(param_grid.keys(), v)) 
        for v in itertools.product(*param_grid.values())
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"Probando: {params}")
        
        model = RandomForestClassifier(**params, random_state=42)
        
        # Validar con TimeSeriesSplit
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        results.append({
            'params': params,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'all_scores': scores
        })
    
    # Ordenar por mejor score
    results = sorted(results, key=lambda x: x['mean_score'], reverse=True)
    
    return results


# Uso
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15]
}

best_params = temporal_grid_search(X_train, y_train, param_grid)
print(f"Mejores parámetros: {best_params[0]}")
```

---

## 🎯 2.3 Regularización para Evitar Overfitting

```python
def train_regularized_model(X_train, y_train):
    """
    Modelo con regularización incorporada.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,              # Limita profundidad (regularización)
        min_samples_split=20,       # Mínimo para dividir nodo
        min_samples_leaf=10,        # Mínimo en hoja
        max_features='sqrt',        # Solo usa sqrt(n_features) por split
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


# Alternativa: XGBoost con regularización explícita
def train_xgboost_regularized(X_train, y_train):
    """
    XGBoost con regularización L1 y L2.
    """
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=0.1,              # Regularización L1
        reg_lambda=1.0,             # Regularización L2
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model
```

---

## 📊 2.4 Feature Selection (Selección de Features)

```python
def select_important_features(model, X, feature_names, threshold=0.01):
    """
    Selecciona solo features importantes.
    Reduce overfitting y mejora velocidad.
    """
    importances = model.feature_importances_
    
    # Crear DataFrame de importancias
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Filtrar por umbral
    selected = importance_df[importance_df['importance'] >= threshold]
    
    print(f"Features seleccionadas: {len(selected)}/{len(feature_names)}")
    print(selected.head(20))
    
    return selected['feature'].tolist()


# Uso
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Seleccionar top features
important_features = select_important_features(
    model, X_train, X_train.columns, threshold=0.01
)

# Reentrenar solo con features importantes
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

model_optimized = RandomForestClassifier(n_estimators=100, random_state=42)
model_optimized.fit(X_train_selected, y_train)
```

---

# 3. REENTRENAMIENTO CONTINUO Y DRIFT DETECTION

## 🎯 Objetivo

Detectar cuando el modelo se degrada y reentrenarlo automáticamente.

---

## 📊 3.1 ¿Qué es Model Drift?

**Concept Drift:** Los patrones que el modelo aprendió cambian con el tiempo.

**Ejemplo:** 
- Modelo entrenado en mercado alcista
- Ahora el mercado es bajista
- Patrones son diferentes → Modelo falla

---

## 🔍 3.2 Detección de Drift

### **A. Monitoreo de Performance**

```python
# ml/drift_detection.py

class PerformanceMonitor:
    """
    Monitorea performance del modelo en tiempo real.
    """
    def __init__(self, window_size=24, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        self.baseline_score = None
    
    def update(self, y_true, y_pred):
        """Actualiza con nueva predicción"""
        from sklearn.metrics import accuracy_score
        
        score = accuracy_score([y_true], [y_pred])
        self.recent_scores.append(score)
        
        # Mantener solo últimas N predicciones
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)
    
    def check_drift(self):
        """Detecta si performance se degradó"""
        if len(self.recent_scores) < self.window_size:
            return False, "Datos insuficientes"
        
        if self.baseline_score is None:
            self.baseline_score = np.mean(self.recent_scores)
            return False, "Baseline establecido"
        
        current_score = np.mean(self.recent_scores)
        degradation = self.baseline_score - current_score
        
        if degradation > self.alert_threshold:
            return True, f"Drift detectado! Degradación: {degradation:.2%}"
        
        return False, f"Performance OK (degradación: {degradation:.2%})"


# Uso
monitor = PerformanceMonitor(window_size=24, alert_threshold=0.10)

# Cada predicción
for y_true, y_pred in predictions:
    monitor.update(y_true, y_pred)
    
    drift_detected, message = monitor.check_drift()
    if drift_detected:
        print(f"ALERTA: {message}")
        # Trigger reentrenamiento
        retrain_model()
```

### **B. Detección Estadística (Kolmogorov-Smirnov)**

```python
from scipy.stats import ks_2samp

def detect_distribution_drift(X_train, X_recent, threshold=0.05):
    """
    Detecta si la distribución de datos cambió.
    Compara features recientes vs datos de entrenamiento.
    """
    drift_features = []
    
    for col in X_train.columns:
        # Test Kolmogorov-Smirnov
        statistic, p_value = ks_2samp(
            X_train[col].dropna(), 
            X_recent[col].dropna()
        )
        
        # Si p-value < threshold → distribuciones son diferentes
        if p_value < threshold:
            drift_features.append({
                'feature': col,
                'p_value': p_value,
                'statistic': statistic
            })
    
    if drift_features:
        print(f"DRIFT DETECTADO en {len(drift_features)} features:")
        for drift in sorted(drift_features, key=lambda x: x['p_value'])[:10]:
            print(f"  {drift['feature']}: p={drift['p_value']:.4f}")
        return True
    
    return False
```

---

## 🔄 3.3 Estrategia de Reentrenamiento

### **Opción A: Reentrenamiento Periódico**

```python
# ml/retraining_scheduler.py

class RetrainingScheduler:
    """
    Reentrena el modelo cada X horas/días.
    """
    def __init__(self, retrain_interval_hours=168):  # 1 semana
        self.retrain_interval_hours = retrain_interval_hours
        self.last_retrain = datetime.now()
    
    def should_retrain(self):
        """Verifica si es tiempo de reentrenar"""
        hours_since_retrain = (
            datetime.now() - self.last_retrain
        ).total_seconds() / 3600
        
        return hours_since_retrain >= self.retrain_interval_hours
    
    def retrain(self, data_loader, model_trainer):
        """Ejecuta reentrenamiento"""
        print("Iniciando reentrenamiento...")
        
        # Cargar datos recientes
        df = data_loader.load_recent_data(days=30)
        
        # Preparar features
        X, y = prepare_features(df)
        
        # Entrenar nuevo modelo
        model = model_trainer.train(X, y)
        
        # Guardar modelo
        model_path = save_model(model, f"model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        
        self.last_retrain = datetime.now()
        
        print(f"Reentrenamiento completado: {model_path}")
        
        return model
```

### **Opción B: Reentrenamiento Adaptativo**

```python
class AdaptiveRetraining:
    """
    Reentrena solo cuando se detecta drift.
    Más eficiente que periódico.
    """
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.drift_detector = detect_distribution_drift
    
    def should_retrain(self, X_train, X_recent, recent_predictions):
        """Decide si reentrenar basado en múltiples señales"""
        
        # Señal 1: Degradación de performance
        drift_detected, _ = self.performance_monitor.check_drift()
        
        # Señal 2: Cambio en distribución de datos
        distribution_drift = self.drift_detector(X_train, X_recent)
        
        # Reentrenar si cualquier señal se activa
        return drift_detected or distribution_drift
```

### **Opción C: Reentrenamiento Incremental (Online Learning)**

```python
from sklearn.linear_model import SGDClassifier

class IncrementalModel:
    """
    Modelo que se actualiza continuamente con nuevos datos.
    Ideal para datos con alta frecuencia.
    """
    def __init__(self):
        self.model = SGDClassifier(
            loss='log_loss',  # Para clasificación
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42
        )
        self.is_fitted = False
    
    def partial_fit(self, X_new, y_new, classes=[0, 1]):
        """Actualiza modelo con nuevos datos"""
        if not self.is_fitted:
            self.model.partial_fit(X_new, y_new, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X_new, y_new)
    
    def predict(self, X):
        return self.model.predict(X)


# Uso
model = IncrementalModel()

# Entrenar con datos iniciales
model.partial_fit(X_initial, y_initial)

# Actualizar cada hora con nuevos datos
for X_hourly, y_hourly in new_data_stream:
    # Primero predecir
    predictions = model.predict(X_hourly)
    
    # Luego actualizar modelo
    model.partial_fit(X_hourly, y_hourly)
```

---

## 📋 3.4 Pipeline de Reentrenamiento Automático

```python
# ml/auto_retrain_pipeline.py

class AutoRetrainPipeline:
    """
    Pipeline completo de reentrenamiento automático.
    """
    def __init__(self, config):
        self.config = config
        self.current_model = None
        self.monitor = PerformanceMonitor()
        self.scheduler = RetrainingScheduler(
            retrain_interval_hours=config['retrain_interval']
        )
    
    def run(self):
        """
        Loop principal de monitoreo y reentrenamiento.
        """
        while True:
            try:
                # 1. Hacer predicción con modelo actual
                X_new = self.load_latest_data()
                y_pred = self.current_model.predict(X_new)
                
                # 2. Esperar resultado real (1 hora después)
                time.sleep(3600)  # 1 hora
                y_true = self.get_actual_result()
                
                # 3. Actualizar monitor
                self.monitor.update(y_true, y_pred)
                
                # 4. Verificar si necesita reentrenar
                drift_detected, message = self.monitor.check_drift()
                time_to_retrain = self.scheduler.should_retrain()
                
                if drift_detected or time_to_retrain:
                    print(f"Reentrenamiento triggered: {message}")
                    self.retrain()
                
            except Exception as e:
                print(f"Error en pipeline: {e}")
                time.sleep(60)
    
    def retrain(self):
        """Ejecuta reentrenamiento"""
        # Cargar datos recientes
        df = self.load_training_data(days=30)
        
        # Preparar features
        X, y = prepare_features(df)
        
        # Entrenar nuevo modelo
        new_model = train_model(X, y)
        
        # Evaluar nuevo modelo
        score = evaluate_model(new_model, X_test, y_test)
        
        # Solo reemplazar si es mejor
        if score > self.monitor.baseline_score:
            self.current_model = new_model
            self.monitor.baseline_score = score
            print(f"Nuevo modelo desplegado (score: {score:.2%})")
        else:
            print(f"Nuevo modelo descartado (score: {score:.2%} vs baseline: {self.monitor.baseline_score:.2%})")
```

---

# 4. EVALUACIÓN SIN DATA LEAKAGE

## 🎯 Objetivo

Medir la precisión real del modelo sin "trucos" ni data leakage.

---

## ⚠️ 4.1 Errores Comunes (Data Leakage)

### **Error 1: Normalizar antes de split**

```python
# ❌ INCORRECTO
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ¡Usa info de test!

X_train, X_test = train_test_split(X_scaled, ...)

# PROBLEMA: El scaler vio los datos de test
```

### **✅ CORRECTO:**

```python
from sklearn.preprocessing import StandardScaler

# Split primero
X_train, X_test = train_test_split(X, ...)

# Normalizar solo en train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Aprende de train

# Aplicar transformación a test
X_test_scaled = scaler.transform(X_test)  # Solo transforma
```

---

### **Error 2: Features que usan datos futuros**

```python
# ❌ INCORRECTO
df['future_return'] = df['price'].shift(-1) / df['price']  # ¡Usa futuro!
df['target'] = (df['future_return'] > 1.01).astype(int)

# PROBLEMA: Conoces el precio futuro cuando predices
```

### **✅ CORRECTO:**

```python
# Solo usar datos PASADOS para features
df['price_lag_1h'] = df['price'].shift(1)  # Pasado
df['price_ma_24h'] = df['price'].rolling(24).mean()  # Pasado

# Target puede usar futuro (pero separado de features)
df['price_future'] = df['price'].shift(-1)
df['target'] = (df['price_future'] > df['price']).astype(int)

# Eliminar última fila (no tiene futuro conocido)
df = df[:-1]
```

---

### **Error 3: Información del test en train**

```python
# ❌ INCORRECTO
df['price_global_mean'] = df['price'].mean()  # ¡Usa todo el dataset!

# PROBLEMA: Incluye información de test en train
```

### **✅ CORRECTO:**

```python
# Calcular estadísticas solo en train
train_mean = df_train['price'].mean()

# Aplicar a train y test
df_train['price_vs_mean'] = df_train['price'] / train_mean
df_test['price_vs_mean'] = df_test['price'] / train_mean  # Usa mean de train
```

---

## ✅ 4.2 Metodología Correcta

### **Pipeline Completo Sin Leakage**

```python
# ml/no_leakage_pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_evaluate_no_leakage(df, test_size=0.2):
    """
    Pipeline completo sin data leakage.
    """
    # 1. Crear features (solo datos pasados)
    df = create_lag_features(df)
    df = create_rolling_mean(df)
    df = create_technical_indicators(df)
    
    # 2. Crear target (puede usar futuro)
    df['price_future'] = df['current_price'].shift(-1)
    df['target'] = (df['price_future'] > df['current_price']).astype(int)
    
    # 3. Eliminar filas con NaNs (por rolling windows)
    df = df.dropna()
    
    # 4. Split temporal (NO aleatorio)
    split_idx = int(len(df) * (1 - test_size))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # 5. Separar X y y
    feature_cols = [col for col in df.columns if col not in ['target', 'price_future', 'last_updated']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    X_test = df_test[feature_cols]
    y_test = df_test['target']
    
    # 6. Pipeline con normalización (aprende solo de train)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 7. Entrenar
    pipeline.fit(X_train, y_train)
    
    # 8. Evaluar
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.2%}")
    print(f"Test accuracy: {test_score:.2%}")
    
    # IMPORTANTE: Test score es la métrica real
    # Si train_score >> test_score → Overfitting
    
    return pipeline, {
        'train_score': train_score,
        'test_score': test_score,
        'overfitting': train_score - test_score
    }
```

---

### **Validación Walk-Forward (La Más Realista)**

```python
def walk_forward_evaluation(df, train_days=30, test_hours=24):
    """
    Simula producción real:
    - Entrena con últimos 30 días
    - Predice próximas 24 horas
    - Reentrena
    - Repite
    """
    results = {
        'predictions': [],
        'actuals': [],
        'timestamps': []
    }
    
    # Ventanas deslizantes
    for i in range(train_days * 24, len(df) - test_hours, test_hours):
        # Train: últimos 30 días
        df_train = df.iloc[i - train_days * 24:i]
        
        # Test: próximas 24 horas
        df_test = df.iloc[i:i + test_hours]
        
        # Preparar datos
        X_train, y_train = prepare_features(df_train)
        X_test, y_test = prepare_features(df_test)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Guardar resultados
        results['predictions'].extend(y_pred)
        results['actuals'].extend(y_test)
        results['timestamps'].extend(df_test['last_updated'].values)
    
    # Evaluar globalmente
    from sklearn.metrics import classification_report
    
    print("Evaluación Walk-Forward:")
    print(classification_report(
        results['actuals'], 
        results['predictions'],
        target_names=['DOWN', 'UP']
    ))
    
    return results
```

---

## 📊 4.3 Métricas de Evaluación

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def comprehensive_evaluation(y_true, y_pred, y_pred_proba=None):
    """
    Evaluación completa del modelo.
    """
    print("="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nMétricas Generales:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1-Score:  {f1:.2%}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de Confusión:")
    print(f"                Pred DOWN  Pred UP")
    print(f"  Actual DOWN   {cm[0,0]:10d}  {cm[0,1]:8d}")
    print(f"  Actual UP     {cm[1,0]:10d}  {cm[1,1]:8d}")
    
    # ROC-AUC (si hay probabilidades)
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        print(f"\nROC-AUC: {auc:.4f}")
    
    # Reporte detallado
    print(f"\nReporte Detallado:")
    print(classification_report(y_true, y_pred, target_names=['DOWN', 'UP']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
```

---

# 5. ESTRATEGIA MULTI-MODELO VS MODELO ÚNICO

## 🎯 Objetivo

Decidir la mejor arquitectura: ¿Un modelo para todo o modelos especializados?

---

## ⚖️ 5.1 Comparación

### **Opción A: Modelo Único (Single Model)**

```
Un solo modelo predice todas las cryptos.

Arquitectura:
┌─────────────────────┐
│  Bitcoin Features   │
│  Ethereum Features  │───▶ [Modelo] ───▶ Predicción
│  Solana Features    │
└─────────────────────┘
```

**Ventajas:**
- ✅ Más simple de mantener
- ✅ Más datos para entrenar (pooling de todas las cryptos)
- ✅ Generaliza mejor a nuevas cryptos
- ✅ Un solo modelo para desplegar

**Desventajas:**
- ❌ No captura características específicas de cada crypto
- ❌ Cryptos muy diferentes (Bitcoin vs meme coins) confunden al modelo
- ❌ Performance promedio puede ser subóptima

**Cuándo usar:**
- Pocas cryptos (<10)
- Datos limitados por crypto
- Cryptos similares (todas top 50)
- Quieres simplicidad

---

### **Opción B: Modelos Especializados (Multi-Model)**

```
Un modelo diferente por cada crypto.

Arquitectura:
Bitcoin Features ───▶ [Modelo Bitcoin]  ───▶ Predicción Bitcoin
Ethereum Features ──▶ [Modelo Ethereum] ───▶ Predicción Ethereum
Solana Features ────▶ [Modelo Solana]   ───▶ Predicción Solana
```

**Ventajas:**
- ✅ Captura características únicas de cada crypto
- ✅ Mejor performance por crypto
- ✅ Parámetros específicos por crypto
- ✅ Falla de un modelo no afecta otros

**Desventajas:**
- ❌ Más complejo de mantener
- ❌ Requiere más datos (por crypto)
- ❌ Más modelos para entrenar/desplegar
- ❌ Más propenso a overfitting (menos datos por modelo)

**Cuándo usar:**
- Muchas cryptos (>20)
- Datos abundantes por crypto (>1000 registros)
- Cryptos muy diferentes (Bitcoin vs meme coins)
- Performance crítica

---

### **Opción C: Híbrido (Recomendado)**

```
Grupos de cryptos similares, un modelo por grupo.

Arquitectura:
Top 10 (Bitcoin, ETH, BNB...) ───▶ [Modelo Top]     ───▶ Predicción
Mid Cap (SOL, ADA, DOT...)    ───▶ [Modelo MidCap]  ───▶ Predicción
Meme Coins (DOGE, SHIB...)    ───▶ [Modelo Meme]    ───▶ Predicción
```

**Ventajas:**
- ✅ Balance entre especialización y generalización
- ✅ Más datos por modelo que enfoque multi-modelo
- ✅ Captura diferencias entre grupos
- ✅ Menos modelos que multi-modelo completo

**Desventajas:**
- ❌ Requiere definir grupos (manual o automático)
- ❌ Más complejo que modelo único

**Cuándo usar:**
- Moderada cantidad de cryptos (10-50)
- Cryptos con características diferentes
- Datos moderados por crypto (>500 registros)
- Quieres balance performance/complejidad

---

## 📊 5.2 Implementación: Modelo Único

```python
# ml/single_model_strategy.py

class SingleModelStrategy:
    """
    Un modelo para todas las cryptos.
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, df_all_cryptos):
        """
        Entrena con datos de todas las cryptos combinadas.
        """
        # Agregar crypto ID como feature (one-hot encoding)
        df = pd.get_dummies(df_all_cryptos, columns=['crypto_id'], prefix='crypto')
        
        # Preparar features
        X, y = self.prepare_features(df)
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def predict(self, df_crypto):
        """
        Predice para cualquier crypto.
        """
        X = self.prepare_features(df_crypto)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
```

---

## 📊 5.3 Implementación: Modelos Especializados

```python
# ml/multi_model_strategy.py

class MultiModelStrategy:
    """
    Un modelo por cada crypto.
    """
    def __init__(self, crypto_ids):
        self.crypto_ids = crypto_ids
        self.models = {}
        self.scalers = {}
    
    def train(self, df_all_cryptos):
        """
        Entrena un modelo por crypto.
        """
        for crypto_id in self.crypto_ids:
            print(f"Entrenando modelo para {crypto_id}...")
            
            # Filtrar datos de esta crypto
            df_crypto = df_all_cryptos[df_all_cryptos['crypto_id'] == crypto_id]
            
            if len(df_crypto) < 100:
                print(f"  Saltando {crypto_id}: datos insuficientes ({len(df_crypto)})")
                continue
            
            # Preparar features
            X, y = self.prepare_features(df_crypto)
            
            # Normalizar (scaler específico por crypto)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Guardar
            self.models[crypto_id] = model
            self.scalers[crypto_id] = scaler
            
            print(f"  ✓ Modelo {crypto_id} entrenado")
        
        return self.models
    
    def predict(self, crypto_id, df_crypto):
        """
        Predice usando el modelo específico de la crypto.
        """
        if crypto_id not in self.models:
            raise ValueError(f"No hay modelo para {crypto_id}")
        
        X = self.prepare_features(df_crypto)
        X_scaled = self.scalers[crypto_id].transform(X)
        
        return self.models[crypto_id].predict(X_scaled)
```

---

## 📊 5.4 Implementación: Híbrido (Recomendado)

```python
# ml/hybrid_model_strategy.py

class HybridModelStrategy:
    """
    Modelos por grupos de cryptos similares.
    """
    def __init__(self):
        self.groups = {
            'top': ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'ripple'],
            'defi': ['uniswap', 'aave', 'compound', 'maker'],
            'meme': ['dogecoin', 'shiba-inu', 'pepe'],
            'stablecoin': ['tether', 'usd-coin', 'dai']
        }
        self.models = {}
        self.scalers = {}
    
    def get_group(self, crypto_id):
        """Identifica a qué grupo pertenece la crypto"""
        for group_name, cryptos in self.groups.items():
            if crypto_id in cryptos:
                return group_name
        return 'other'  # Grupo por defecto
    
    def train(self, df_all_cryptos):
        """
        Entrena un modelo por grupo.
        """
        for group_name in self.groups.keys():
            print(f"Entrenando modelo para grupo: {group_name}")
            
            # Filtrar cryptos de este grupo
            group_cryptos = self.groups[group_name]
            df_group = df_all_cryptos[df_all_cryptos['crypto_id'].isin(group_cryptos)]
            
            if len(df_group) < 200:
                print(f"  Saltando {group_name}: datos insuficientes")
                continue
            
            # Preparar features (incluir crypto_id como feature)
            df_group = pd.get_dummies(df_group, columns=['crypto_id'], prefix='crypto')
            X, y = self.prepare_features(df_group)
            
            # Normalizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_scaled, y)
            
            # Guardar
            self.models[group_name] = model
            self.scalers[group_name] = scaler
            
            print(f"  ✓ Modelo {group_name} entrenado ({len(df_group)} registros)")
        
        return self.models
    
    def predict(self, crypto_id, df_crypto):
        """
        Predice usando el modelo del grupo correspondiente.
        """
        group = self.get_group(crypto_id)
        
        if group not in self.models:
            raise ValueError(f"No hay modelo para el grupo: {group}")
        
        # Preparar features
        df_crypto = pd.get_dummies(df_crypto, columns=['crypto_id'], prefix='crypto')
        X = self.prepare_features(df_crypto)
        
        # Normalizar
        X_scaled = self.scalers[group].transform(X)
        
        # Predecir
        return self.models[group].predict(X_scaled)
```

---

## 🎯 5.5 Comparación Experimental

```python
# ml/compare_strategies.py

def compare_strategies(df_all_cryptos):
    """
    Compara las 3 estrategias y recomienda la mejor.
    """
    results = {}
    
    # Split temporal
    split_idx = int(len(df_all_cryptos) * 0.8)
    df_train = df_all_cryptos.iloc[:split_idx]
    df_test = df_all_cryptos.iloc[split_idx:]
    
    # 1. Modelo único
    print("1. Evaluando Single Model...")
    single = SingleModelStrategy()
    single.train(df_train)
    
    predictions_single = []
    actuals_single = []
    for crypto_id in df_test['crypto_id'].unique():
        df_crypto_test = df_test[df_test['crypto_id'] == crypto_id]
        pred = single.predict(df_crypto_test)
        predictions_single.extend(pred)
        actuals_single.extend(df_crypto_test['target'])
    
    results['single'] = accuracy_score(actuals_single, predictions_single)
    print(f"   Accuracy: {results['single']:.2%}")
    
    # 2. Multi-modelo
    print("\n2. Evaluando Multi-Model...")
    multi = MultiModelStrategy(df_train['crypto_id'].unique())
    multi.train(df_train)
    
    predictions_multi = []
    actuals_multi = []
    for crypto_id in df_test['crypto_id'].unique():
        if crypto_id not in multi.models:
            continue
        df_crypto_test = df_test[df_test['crypto_id'] == crypto_id]
        pred = multi.predict(crypto_id, df_crypto_test)
        predictions_multi.extend(pred)
        actuals_multi.extend(df_crypto_test['target'])
    
    results['multi'] = accuracy_score(actuals_multi, predictions_multi)
    print(f"   Accuracy: {results['multi']:.2%}")
    
    # 3. Híbrido
    print("\n3. Evaluando Hybrid...")
    hybrid = HybridModelStrategy()
    hybrid.train(df_train)
    
    predictions_hybrid = []
    actuals_hybrid = []
    for crypto_id in df_test['crypto_id'].unique():
        df_crypto_test = df_test[df_test['crypto_id'] == crypto_id]
        try:
            pred = hybrid.predict(crypto_id, df_crypto_test)
            predictions_hybrid.extend(pred)
            actuals_hybrid.extend(df_crypto_test['target'])
        except ValueError:
            continue
    
    results['hybrid'] = accuracy_score(actuals_hybrid, predictions_hybrid)
    print(f"   Accuracy: {results['hybrid']:.2%}")
    
    # Resultado
    print("\n" + "="*60)
    print("RESULTADO:")
    print("="*60)
    best_strategy = max(results, key=results.get)
    print(f"Mejor estrategia: {best_strategy.upper()} ({results[best_strategy]:.2%})")
    
    return results
```

---

## 🎯 5.6 Recomendación (Basada en tus datos)

Según tu situación actual:

```
Datos actuales: 3,484 registros × 262 cryptos
Promedio por crypto: ~13 registros
```

**Recomendación: MODELO ÚNICO (Single Model)**

**Razones:**
1. ✅ **Pocos datos por crypto** (13 registros promedio)
2. ✅ **Muchas cryptos** (262) → Multi-modelo no es viable aún
3. ✅ **Simplicidad** → Más fácil de mantener al principio
4. ✅ **Generalización** → Aprende patrones comunes

**Cuándo cambiar a Híbrido:**
- Cuando tengas >500 registros por crypto (en ~2 meses)
- Empieza con grupos: Top 20, Mid Cap, Meme Coins, Stablecoins

**Cuándo cambiar a Multi-Model:**
- Cuando tengas >1,000 registros por crypto (en ~3 meses)
- Solo para cryptos top (Bitcoin, Ethereum, etc.)

---

# 📋 RESUMEN EJECUTIVO

## ✅ Checklist de Implementación

### **Fase 1: Feature Engineering (Esta semana)**
- [ ] Implementar lags (1h, 3h, 6h, 12h, 24h)
- [ ] Implementar rolling means (6h, 12h, 24h)
- [ ] Implementar RSI (14h, 24h)
- [ ] Implementar MACD
- [ ] Implementar features de volatilidad
- [ ] Implementar features temporales (hora, día)
- [ ] Tests para cada tipo de feature

### **Fase 2: Optimización (Próxima semana)**
- [ ] Implementar Time Series Split
- [ ] Grid search con validación temporal
- [ ] Feature selection (top 30 features)
- [ ] Regularización (max_depth, min_samples)
- [ ] Walk-forward validation

### **Fase 3: Reentrenamiento (En 2 semanas)**
- [ ] Performance monitor
- [ ] Drift detection (estadístico)
- [ ] Scheduler de reentrenamiento
- [ ] Pipeline automático

### **Fase 4: Evaluación (Continuo)**
- [ ] Eliminar data leakage
- [ ] Pipeline sin leakage
- [ ] Métricas comprehensivas
- [ ] Validación walk-forward

### **Fase 5: Estrategia Multi-Modelo (En 1 mes)**
- [ ] Empezar con modelo único
- [ ] Evaluar performance por crypto
- [ ] Cuando haya datos, probar híbrido
- [ ] Comparar estrategias

---

## 🎯 Prioridades Inmediatas

**Esta semana:**
1. ✅ Implementar features avanzadas (lags, RSI, MACD)
2. ✅ Validación temporal (Time Series Split)
3. ✅ Eliminar data leakage del pipeline actual

**Próxima semana:**
1. Grid search temporal
2. Feature selection
3. Performance monitor básico

**En 1 mes:**
1. Reentrenamiento automático
2. Drift detection
3. Evaluar estrategia híbrida

---

## 📚 Recursos Adicionales

- **Feature Engineering:** "Feature Engineering for Machine Learning" (Casari & Zheng)
- **Time Series:** "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- **Model Drift:** "Evidently AI" (librería Python para drift detection)
- **Trading ML:** "Advances in Financial Machine Learning" (Marcos López de Prado)

---

**Esta guía es tu roadmap completo de "funcional" a "producción".** 🚀

¿Quieres que empiece a implementar alguna de estas fases específicamente?

