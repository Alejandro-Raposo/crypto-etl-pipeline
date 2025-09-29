# 🚀 Configuración de GitHub Actions para Crypto ML Pipeline

## Prerrequisitos
- Repositorio en GitHub (público o privado)
- Credenciales de Google Cloud (archivo JSON)

## Paso 1: Subir el proyecto a GitHub

```bash
# Si no tienes el repo creado
git init
git add .
git commit -m "Initial commit: Crypto ML Pipeline"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/crypto-etl-pipeline.git
git push -u origin main
```

## Paso 2: Configurar Secretos en GitHub

1. Ve a tu repositorio en GitHub
2. Clic en **Settings** > **Secrets and variables** > **Actions**
3. Clic en **New repository secret**
4. Crear el secreto:
   - **Name**: `GOOGLE_CREDENTIALS`
   - **Value**: Pegar todo el contenido del archivo JSON de credenciales

## Paso 3: Verificar el Workflow

El archivo `.github/workflows/crypto-ml-pipeline.yml` ya está configurado para:

- ✅ **Ejecutarse cada hora** (0 * * * *)
- ✅ **Instalar dependencias** automáticamente
- ✅ **Ejecutar el pipeline completo**
- ✅ **Guardar logs y datos** como artefactos
- ✅ **Mostrar resumen** de ejecución

## Paso 4: Ejecutar Manualmente (Primera vez)

1. Ve a **Actions** en tu repositorio
2. Selecciona **Crypto ML Data Pipeline**
3. Clic en **Run workflow** > **Run workflow**

## Paso 5: Monitorear Ejecuciones

- **Logs en tiempo real**: Actions > Crypto ML Data Pipeline > [Última ejecución]
- **Artefactos descargables**: Datos procesados y logs
- **Notificaciones**: Email automático si falla

## Costos

### Repositorio PÚBLICO: 
- ✅ **GRATIS ILIMITADO**

### Repositorio PRIVADO:
- ✅ **2,000 minutos gratis/mes**
- ✅ **Pipeline tarda ~2 minutos = 1,440 ejecuciones/mes gratis**
- ✅ **Más que suficiente para ejecución horaria**

## Alternativas de Configuración

### Ejecución cada 2 horas:
```yaml
schedule:
  - cron: '0 */2 * * *'
```

### Solo días laborables:
```yaml
schedule:
  - cron: '0 9-17 * * 1-5'  # 9 AM - 5 PM, Lun-Vie
```

### Solo horarios de mercado (UTC):
```yaml
schedule:
  - cron: '0 8-20 * * 1-5'  # 8 AM - 8 PM UTC, Lun-Vie
```

## Troubleshooting

### Error de credenciales:
- Verificar que `GOOGLE_CREDENTIALS` está configurado correctamente
- El contenido debe ser el JSON completo, incluyendo las llaves {}

### Error de permisos BigQuery:
- Verificar que la Service Account tiene permisos de BigQuery Data Editor
- Verificar que el proyecto ID es correcto

### Pipeline falla:
- Revisar logs en Actions > [Ejecución fallida] > [Step fallido]
- Los errores más comunes aparecen en el step "Run ML Pipeline"

## Ventajas de GitHub Actions

✅ **Cero configuración de infraestructura**
✅ **Logs detallados y notificaciones**
✅ **Integración con Git (versionado)**
✅ **Escalable y confiable**
✅ **Gratis para uso personal**

## Siguientes Pasos

Una vez configurado, tendrás:
- 📊 **Datos recolectados cada hora automáticamente**
- 📈 **Histórico creciendo en BigQuery**
- 🔍 **Monitoreo y alertas automáticas**
- 📁 **Backups en GitHub Artifacts**

En **1 semana = 168 snapshots** listos para entrenar modelos ML! 🤖
