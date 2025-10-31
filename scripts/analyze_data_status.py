"""
Análisis completo del estado actual de los datos en BigQuery.
Determina qué mejoras de ML podemos implementar según los datos disponibles.
"""

import os
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
PROJECT = os.getenv('GCP_PROJECT_ID')
DATASET = os.getenv('BIGQUERY_DATASET_ID')
TABLE_HISTORICAL = 'prices_historical'
TABLE_ML_FEATURES = 'prices_ml_features'

def analyze_data_status():
    """
    Analiza el estado actual de los datos y recomienda próximos pasos.
    """
    print("=" * 80)
    print("ANALISIS DEL ESTADO ACTUAL DE DATOS")
    print("=" * 80)
    
    # Configurar cliente BigQuery
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        client = bigquery.Client(credentials=credentials, project=PROJECT)
    else:
        client = bigquery.Client(project=PROJECT)
    
    # ===================================================================
    # 1. ANÁLISIS DE DATOS HISTÓRICOS (prices_historical)
    # ===================================================================
    print("\n1. TABLA: prices_historical")
    print("-" * 80)
    
    query_historical = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as total_cryptos,
        MIN(last_updated) as first_record,
        MAX(last_updated) as last_record,
        TIMESTAMP_DIFF(MAX(last_updated), MIN(last_updated), HOUR) as hours_coverage
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    """
    
    try:
        df_hist = client.query(query_historical).to_dataframe()
        
        total_records = df_hist['total_records'].iloc[0]
        total_cryptos = df_hist['total_cryptos'].iloc[0]
        first_record = df_hist['first_record'].iloc[0]
        last_record = df_hist['last_record'].iloc[0]
        hours_coverage = df_hist['hours_coverage'].iloc[0]
        
        print(f"Total de registros:     {total_records:,}")
        print(f"Total de cryptos:       {total_cryptos}")
        print(f"Primer registro:        {first_record}")
        print(f"Ultimo registro:        {last_record}")
        print(f"Cobertura temporal:     {hours_coverage:,} horas ({hours_coverage/24:.1f} dias)")
        print(f"Promedio por crypto:    {total_records/total_cryptos:.0f} registros")
        
    except Exception as e:
        print(f"Error al analizar prices_historical: {e}")
        return
    
    # ===================================================================
    # 2. DISTRIBUCIÓN POR CRYPTO (Top 20)
    # ===================================================================
    print("\n2. TOP 20 CRYPTOS (por cantidad de registros)")
    print("-" * 80)
    
    query_top_cryptos = f"""
    SELECT 
        id,
        COUNT(*) as num_records,
        MIN(last_updated) as first_seen,
        MAX(last_updated) as last_seen,
        TIMESTAMP_DIFF(MAX(last_updated), MIN(last_updated), HOUR) as hours_span
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    GROUP BY id
    ORDER BY num_records DESC
    LIMIT 20
    """
    
    try:
        df_top = client.query(query_top_cryptos).to_dataframe()
        
        print(f"{'Crypto':<20} {'Registros':>12} {'Cobertura (dias)':>18} {'Avg/dia':>10}")
        print("-" * 80)
        
        for _, row in df_top.iterrows():
            avg_per_day = row['num_records'] / (row['hours_span'] / 24) if row['hours_span'] > 0 else 0
            print(f"{row['id']:<20} {row['num_records']:>12,} {row['hours_span']/24:>17.1f} {avg_per_day:>10.1f}")
        
    except Exception as e:
        print(f"Error al analizar top cryptos: {e}")
    
    # ===================================================================
    # 3. ANÁLISIS DE FEATURES ML (prices_ml_features)
    # ===================================================================
    print("\n3. TABLA: prices_ml_features")
    print("-" * 80)
    
    query_ml_features = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as total_cryptos,
        MIN(last_updated) as first_record,
        MAX(last_updated) as last_record
    FROM `{PROJECT}.{DATASET}.{TABLE_ML_FEATURES}`
    """
    
    try:
        df_ml = client.query(query_ml_features).to_dataframe()
        
        ml_total = df_ml['total_records'].iloc[0]
        ml_cryptos = df_ml['total_cryptos'].iloc[0]
        ml_first = df_ml['first_record'].iloc[0]
        ml_last = df_ml['last_record'].iloc[0]
        
        print(f"Total de registros:     {ml_total:,}")
        print(f"Total de cryptos:       {ml_cryptos}")
        print(f"Primer registro:        {ml_first}")
        print(f"Ultimo registro:        {ml_last}")
        print(f"Promedio por crypto:    {ml_total/ml_cryptos:.0f} registros")
        
    except Exception as e:
        print(f"Error al analizar prices_ml_features: {e}")
        ml_total = 0
    
    # ===================================================================
    # 4. CALIDAD DE DATOS
    # ===================================================================
    print("\n4. CALIDAD DE DATOS (ultimas 24 horas)")
    print("-" * 80)
    
    query_quality = f"""
    SELECT 
        COUNT(*) as records_24h,
        COUNT(DISTINCT id) as cryptos_24h,
        COUNT(DISTINCT EXTRACT(HOUR FROM last_updated)) as unique_hours
    FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
    WHERE last_updated >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    """
    
    try:
        df_quality = client.query(query_quality).to_dataframe()
        
        records_24h = df_quality['records_24h'].iloc[0]
        cryptos_24h = df_quality['cryptos_24h'].iloc[0]
        unique_hours = df_quality['unique_hours'].iloc[0]
        
        print(f"Registros en 24h:       {records_24h:,}")
        print(f"Cryptos activas:        {cryptos_24h}")
        print(f"Horas con datos:        {unique_hours}/24")
        
        if unique_hours >= 20:
            print(f"Estado:                 EXCELENTE (cobertura {unique_hours/24*100:.0f}%)")
        elif unique_hours >= 15:
            print(f"Estado:                 BUENO (cobertura {unique_hours/24*100:.0f}%)")
        else:
            print(f"Estado:                 MEJORABLE (cobertura {unique_hours/24*100:.0f}%)")
        
    except Exception as e:
        print(f"Error al analizar calidad: {e}")
    
    # ===================================================================
    # 5. GAPS EN LOS DATOS
    # ===================================================================
    print("\n5. DETECCION DE GAPS (Bitcoin como referencia)")
    print("-" * 80)
    
    query_gaps = f"""
    WITH bitcoin_data AS (
        SELECT 
            last_updated,
            TIMESTAMP_DIFF(
                last_updated, 
                LAG(last_updated) OVER (ORDER BY last_updated), 
                HOUR
            ) as hours_gap
        FROM `{PROJECT}.{DATASET}.{TABLE_HISTORICAL}`
        WHERE id = 'bitcoin'
        ORDER BY last_updated DESC
        LIMIT 100
    )
    SELECT 
        COUNT(*) as total_intervals,
        AVG(hours_gap) as avg_gap_hours,
        MAX(hours_gap) as max_gap_hours,
        COUNTIF(hours_gap > 2) as gaps_over_2h
    FROM bitcoin_data
    WHERE hours_gap IS NOT NULL
    """
    
    try:
        df_gaps = client.query(query_gaps).to_dataframe()
        
        avg_gap = df_gaps['avg_gap_hours'].iloc[0]
        max_gap = df_gaps['max_gap_hours'].iloc[0]
        gaps_over_2h = df_gaps['gaps_over_2h'].iloc[0]
        
        print(f"Intervalo promedio:     {avg_gap:.2f} horas")
        print(f"Gap maximo:             {max_gap:.0f} horas")
        print(f"Gaps > 2 horas:         {gaps_over_2h}")
        
        if avg_gap <= 1.5:
            print(f"Frecuencia:             OPTIMA (aprox. cada hora)")
        elif avg_gap <= 3:
            print(f"Frecuencia:             BUENA")
        else:
            print(f"Frecuencia:             MEJORABLE")
        
    except Exception as e:
        print(f"Error al analizar gaps: {e}")
    
    # ===================================================================
    # 6. RECOMENDACIONES BASADAS EN DATOS
    # ===================================================================
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PARA PROXIMOS PASOS")
    print("=" * 80)
    
    avg_records_per_crypto = total_records / total_cryptos
    days_coverage = hours_coverage / 24
    
    recommendations = []
    
    # Evaluar qué se puede implementar
    if avg_records_per_crypto >= 1000 and days_coverage >= 30:
        recommendations.append({
            'priority': 'ALTA',
            'fase': 'FASE 1 - Feature Engineering Avanzado',
            'accion': 'Implementar features avanzadas (lags, RSI, MACD, volatilidad)',
            'motivo': f'Tienes {avg_records_per_crypto:.0f} registros/crypto y {days_coverage:.1f} dias de datos'
        })
    
    if avg_records_per_crypto >= 500 and days_coverage >= 20:
        recommendations.append({
            'priority': 'ALTA',
            'fase': 'FASE 1 - Validacion Temporal',
            'accion': 'Implementar Time Series Split y Walk-Forward Validation',
            'motivo': f'Suficientes datos para validacion robusta'
        })
    
    if avg_records_per_crypto >= 2000 and ml_total > 0:
        recommendations.append({
            'priority': 'MEDIA',
            'fase': 'FASE 2 - Optimizacion',
            'accion': 'Grid Search temporal y Feature Selection',
            'motivo': f'Datos suficientes para optimizacion de hiperparametros'
        })
    
    if avg_records_per_crypto >= 500:
        recommendations.append({
            'priority': 'ALTA',
            'fase': 'FASE 3 - Monitoring',
            'accion': 'Implementar Performance Monitor y Drift Detection',
            'motivo': f'Datos suficientes para monitoreo continuo'
        })
    
    if avg_records_per_crypto >= 1000 and total_cryptos >= 50:
        recommendations.append({
            'priority': 'MEDIA',
            'fase': 'FASE 4 - Estrategia Multi-Modelo',
            'accion': 'Evaluar modelo Hibrido (grupos de cryptos)',
            'motivo': f'Tienes {total_cryptos} cryptos con {avg_records_per_crypto:.0f} registros/crypto'
        })
    
    if avg_records_per_crypto < 500:
        recommendations.append({
            'priority': 'ALTA',
            'fase': 'ESPERAR',
            'accion': 'Continuar acumulando datos',
            'motivo': f'Solo {avg_records_per_crypto:.0f} registros/crypto. Objetivo: >500'
        })
    
    # Mostrar recomendaciones
    if recommendations:
        print("\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['priority']}] {rec['fase']}")
            print(f"   Accion: {rec['accion']}")
            print(f"   Motivo: {rec['motivo']}")
            print()
    
    # ===================================================================
    # 7. SUMMARY
    # ===================================================================
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    print(f"\nDATOS DISPONIBLES:")
    print(f"  - {total_records:,} registros totales")
    print(f"  - {total_cryptos} cryptos")
    print(f"  - {avg_records_per_crypto:.0f} registros/crypto (promedio)")
    print(f"  - {days_coverage:.1f} dias de cobertura")
    
    print(f"\nESTADO DEL SISTEMA:")
    if avg_records_per_crypto >= 1000:
        print(f"  EXCELENTE - Listo para implementar features avanzadas")
    elif avg_records_per_crypto >= 500:
        print(f"  BUENO - Puedes empezar con features basicas")
    elif avg_records_per_crypto >= 200:
        print(f"  ACEPTABLE - Continua acumulando datos")
    else:
        print(f"  INSUFICIENTE - Necesitas mas datos (objetivo: >500/crypto)")
    
    print(f"\nPROXIMO PASO RECOMENDADO:")
    if recommendations:
        top_rec = recommendations[0]
        print(f"  {top_rec['accion']}")
        print(f"  Razon: {top_rec['motivo']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        analyze_data_status()
    except Exception as e:
        print(f"Error en analisis: {e}")
        import traceback
        traceback.print_exc()

