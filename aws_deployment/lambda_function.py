"""
AWS Lambda function para ejecutar el pipeline de ML de criptomonedas.
"""

import json
import boto3
import os
import tempfile
from pathlib import Path

def lambda_handler(event, context):
    """
    AWS Lambda handler para el pipeline de criptomonedas.
    """
    try:
        print("🚀 Starting Crypto ML Pipeline on AWS Lambda...")
        
        # Configurar directorios temporales
        temp_dir = Path("/tmp/crypto-pipeline")
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(temp_dir)
        
        # Importar módulos del pipeline
        import sys
        sys.path.append(str(Path(__file__).parent))
        
        from extract import fetch_coingecko_top, add_extraction_metadata
        from transform import transform
        
        # EXTRACT
        print("📥 Extracting data...")
        raw_data = fetch_coingecko_top(n=250)
        data_with_metadata = add_extraction_metadata(raw_data)
        
        # TRANSFORM
        print("🔄 Transforming data...")
        df = transform(data_with_metadata["data"], data_with_metadata["metadata"])
        
        # LOAD - Guardar en S3 y/o BigQuery
        print("📤 Loading data...")
        
        # Opción 1: Guardar en S3
        import pandas as pd
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('S3_BUCKET', 'crypto-ml-data')
        
        # Convertir a Parquet en memoria
        parquet_buffer = df.to_parquet(index=False)
        
        timestamp = data_with_metadata["metadata"]["extraction_timestamp"]
        s3_key = f"processed/{timestamp}/crypto_data.parquet"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=parquet_buffer
        )
        
        result = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Pipeline completed successfully',
                'rows_processed': len(df),
                'features_count': len(df.columns),
                's3_location': f's3://{bucket_name}/{s3_key}',
                'timestamp': timestamp
            })
        }
        
        print(f"✅ Success: {result}")
        return result
        
    except Exception as e:
        error_result = {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Pipeline execution failed'
            })
        }
        print(f"❌ Error: {error_result}")
        return error_result
