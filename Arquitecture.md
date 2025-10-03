# Directrices de Arquitectura del Proyecto

**Reglas obligatorias** para cualquier implementación en este repositorio.

---

## 🎯 **1. RENDIMIENTO Y EFICIENCIA**

**Regla**: Siempre buscar la solución más eficiente en términos de rendimiento.

**Implica**:
- Optimizar queries de base de datos
- Usar estructuras de datos apropiadas
- Evitar loops innecesarios
- Aprovechar operaciones vectorizadas (pandas, numpy)

---

## 🧹 **2. CÓDIGO LIMPIO**

### **Sin prints innecesarios**
- ❌ No usar `print()` para debugging
- ✅ Usar `logging` para información importante
- ✅ Solo logs relevantes para el usuario final

### **Sin comentarios innecesarios**
- ❌ No comentar código obvio
- ✅ Documentar lógica compleja
- ✅ Usar docstrings en funciones

**Ejemplo Malo**:
```python
# Sumar a y b
resultado = a + b  # suma los números
print(resultado)  # imprime el resultado
```

**Ejemplo Bueno**:
```python
def calcular_media_movil(precios, ventana=24):
    """
    Calcula media móvil de precios.
    Args:
        precios: Serie de precios históricos
        ventana: Número de períodos para el cálculo
    Returns:
        Serie con media móvil calculada
    """
    return precios.rolling(window=ventana).mean()
```

---

## 📐 **3. ESPACIADO Y FORMATO**

**Regla**: Máximo 1 línea en blanco entre bloques de código.

**Ejemplo Malo**:
```python
def funcion1():
    pass


def funcion2():
    pass
```

**Ejemplo Bueno**:
```python
def funcion1():
    pass

def funcion2():
    pass
```

---

## 📏 **4. TAMAÑO DE FUNCIONES**

**Regla**: Las funciones NO deben ser más largas que un scroll de pantalla (~50 líneas máximo).

**Si una función es muy larga**:
- ✅ Dividirla en subfunciones
- ✅ Extraer lógica a funciones auxiliares
- ✅ Mantener una responsabilidad por función

**Ejemplo**:
```python
# ❌ MAL: Función de 150 líneas
def process_all_data():
    # ... 150 líneas de código

# ✅ BIEN: Dividida en funciones pequeñas
def process_all_data():
    data = load_data()
    data = clean_data(data)
    data = transform_data(data)
    return save_data(data)
```

---

## 🔄 **5. REUTILIZACIÓN DE CÓDIGO (DRY - Don't Repeat Yourself)**

**Regla**: Si una funcionalidad se necesita en múltiples archivos, crearla UNA VEZ e importarla.

**Estructura**:
```
scripts/
├── utils/
│   ├── database.py      # Funciones de conexión DB
│   ├── validators.py    # Validaciones reutilizables
│   └── helpers.py       # Helpers generales
├── extract.py           # Importa de utils/
├── transform.py         # Importa de utils/
└── load_historical.py   # Importa de utils/
```

**Ejemplo**:
```python
# utils/database.py
def get_bigquery_client():
    """Retorna cliente de BigQuery configurado"""
    return bigquery.Client.from_service_account_json(credentials_path)

# Usar en múltiples archivos
from utils.database import get_bigquery_client
client = get_bigquery_client()
```

---

## 📦 **6. ORGANIZACIÓN DE IMPORTS**

**Regla**: TODOS los imports al inicio del archivo, organizados.

**Orden correcto**:
```python
# 1. Imports de librería estándar
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# 2. Imports de terceros
import pandas as pd
import numpy as np
from google.cloud import bigquery

# 3. Imports locales
from utils.database import get_client
from utils.validators import validate_price
```

---

## ✅ **7. TEST-DRIVEN DEVELOPMENT (TDD)** 🔴 **REGLA MÁS IMPORTANTE**

**Regla CRÍTICA**: Los tests se DISEÑAN y ESCRIBEN **ANTES** que la funcionalidad.

### **⚠️ ORDEN OBLIGATORIO (NO NEGOCIABLE)**:

```
❌ INCORRECTO:
   Implementar funcionalidad → Escribir tests

✅ CORRECTO:
   1. Diseñar tests estrictos
   2. Implementar funcionalidad que pase esos tests
```

### **🎯 Filosofía TDD**:

Los tests NO son validación opcional. Son la **especificación exacta** de cómo debe funcionar el código.

- **Tests = Contrato**: Definen qué debe hacer el código
- **Código = Implementación**: Debe cumplir ese contrato al 100%
- **Tests estrictos** = Código robusto desde el principio
- **Sin tests primero** = Código que probablemente falla en producción

### **Proceso Obligatorio**:

1. **Entender la funcionalidad** que se va a implementar
2. **DISEÑAR TESTS ESTRICTOS** en `test/` que definan el comportamiento esperado
   - Tests deben ser **imposibles de pasar** con código incorrecto
   - Deben cubrir **todos los casos**: normales, límite, errores
3. **Ejecutar tests** (deben fallar en rojo 🔴 porque aún no existe la funcionalidad)
4. **Implementar funcionalidad** ESPECÍFICAMENTE para pasar esos tests
5. **Ejecutar tests** (deben pasar en verde ✅)
6. **Refactorizar** si es necesario (manteniendo tests en verde)

### **⚡ Beneficio Clave**:

Implementar código que pasa tests estrictos diseñados previamente **GARANTIZA**:
- ✅ Funcionamiento correcto desde el primer momento
- ✅ No bugs en producción
- ✅ Código que hace exactamente lo que debe hacer
- ✅ Confianza total en cada implementación

### **Requisitos de Tests**:
- ✅ Tests deben ser **imposibles de pasar** sin implementar correctamente
- ✅ Deben cubrir **casos normales** y **casos extremos**
- ✅ Deben ser **específicos** y **claros**
- ✅ Cada test debe probar **UNA cosa**

### **Ejemplo Completo**:

**Paso 1: Escribir test PRIMERO** (`test/test_validacion_precio.py`)
```python
def test_debe_rechazar_precios_negativos():
    data = {'id': 'bitcoin', 'current_price': -100}
    df = pd.DataFrame([data])
    df_clean = validar_precios(df)
    assert len(df_clean) == 0, "Debe eliminar precios negativos"

def test_debe_rechazar_precios_nulos():
    data = {'id': 'bitcoin', 'current_price': None}
    df = pd.DataFrame([data])
    df_clean = validar_precios(df)
    assert len(df_clean) == 0, "Debe eliminar precios nulos"

def test_debe_mantener_precios_validos():
    data = {'id': 'bitcoin', 'current_price': 62000}
    df = pd.DataFrame([data])
    df_clean = validar_precios(df)
    assert len(df_clean) == 1, "Debe mantener precios válidos"
```

**Paso 2: Implementar funcionalidad** (`scripts/utils/validators.py`)
```python
def validar_precios(df):
    """
    Valida y filtra precios inválidos.
    Elimina precios negativos, nulos o cero.
    """
    df = df.dropna(subset=['current_price'])
    df = df[df['current_price'] > 0]
    return df
```

**Paso 3: Ejecutar tests**
```bash
pytest test/test_validacion_precio.py -v
# Todos deben pasar ✅
```

---

## 📋 **CHECKLIST ANTES DE COMMIT**

Antes de hacer commit, verificar:

- [ ] ✅ Código eficiente y optimizado
- [ ] ✅ Sin prints innecesarios
- [ ] ✅ Sin comentarios obvios
- [ ] ✅ Máximo 1 línea en blanco entre bloques
- [ ] ✅ Funciones < 50 líneas
- [ ] ✅ Código reutilizable extraído a utils/
- [ ] ✅ Imports organizados al inicio
- [ ] ✅ **Tests escritos y pasando**
- [ ] ✅ Tests cubren casos normales y extremos

---

## 🎯 **RESUMEN EJECUTIVO**

| # | Regla | Prioridad |
|---|-------|-----------|
| 1 | Rendimiento óptimo | 🔴 CRÍTICA |
| 2 | Código limpio (sin prints/comentarios innecesarios) | 🟡 ALTA |
| 3 | Máximo 1 espacio entre líneas | 🟢 MEDIA |
| 4 | Funciones cortas (< 50 líneas) | 🟡 ALTA |
| 5 | DRY - No repetir código | 🟡 ALTA |
| 6 | Imports al inicio, organizados | 🟢 MEDIA |
| 7 | **TDD - Tests antes de código** | 🔴 **CRÍTICA** |

---

**Estas reglas son OBLIGATORIAS** y deben seguirse en TODAS las implementaciones del proyecto.
