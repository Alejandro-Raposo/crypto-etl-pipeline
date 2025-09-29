#!/bin/bash

# Script para desplegar la Cloud Function de ML Pipeline

echo "🚀 Desplegando Crypto ML Pipeline en Google Cloud..."

# Variables de configuración
FUNCTION_NAME="crypto-ml-pipeline"
REGION="europe-west1"
PROJECT_ID="crypto-etl-proyect"
RUNTIME="python311"
TRIGGER="--trigger-http"
MEMORY="512MB"
TIMEOUT="540s"

# Copiar archivos del pipeline
echo "📁 Copiando archivos del pipeline..."
cp -r ../scripts ./
cp ../.env ./ 2>/dev/null || echo "No .env file found"

# Desplegar función
echo "☁️ Desplegando Cloud Function..."
gcloud functions deploy $FUNCTION_NAME \
    --runtime $RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --memory $MEMORY \
    --timeout $TIMEOUT \
    --region $REGION \
    --project $PROJECT_ID \
    --entry-point crypto_ml_pipeline \
    --set-env-vars "GCP_PROJECT=$PROJECT_ID"

# Crear Cloud Scheduler job
echo "⏰ Configurando Cloud Scheduler..."
gcloud scheduler jobs create http crypto-ml-hourly \
    --schedule="0 * * * *" \
    --uri="https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME" \
    --http-method=GET \
    --location=$REGION \
    --project=$PROJECT_ID \
    --description="Crypto ML Pipeline - Hourly execution"

echo "✅ Despliegue completado!"
echo "🔗 URL de la función: https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME"
echo "📊 Monitor: https://console.cloud.google.com/functions/details/$REGION/$FUNCTION_NAME"
