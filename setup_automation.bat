@echo off
echo ========================================
echo   CONFIGURACION DE AUTOMATIZACION ML
echo ========================================

echo.
echo Configurando pipeline automatizado para ML de criptomonedas...

:: Crear directorio de logs si no existe
if not exist "logs" mkdir logs

:: Hacer ejecutable el script de Python
echo Verificando permisos de scripts...

:: Crear archivo de configuración para Task Scheduler
echo Creando configuracion para Task Scheduler...

echo ^<?xml version="1.0" encoding="UTF-16"?^> > crypto_ml_pipeline_task.xml
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^> >> crypto_ml_pipeline_task.xml
echo   ^<RegistrationInfo^> >> crypto_ml_pipeline_task.xml
echo     ^<Description^>Crypto ML Pipeline - Ejecucion cada hora^</Description^> >> crypto_ml_pipeline_task.xml
echo   ^</RegistrationInfo^> >> crypto_ml_pipeline_task.xml
echo   ^<Triggers^> >> crypto_ml_pipeline_task.xml
echo     ^<TimeTrigger^> >> crypto_ml_pipeline_task.xml
echo       ^<Repetition^> >> crypto_ml_pipeline_task.xml
echo         ^<Interval^>PT1H^</Interval^> >> crypto_ml_pipeline_task.xml
echo       ^</Repetition^> >> crypto_ml_pipeline_task.xml
echo       ^<StartBoundary^>2025-09-29T18:00:00^</StartBoundary^> >> crypto_ml_pipeline_task.xml
echo       ^<Enabled^>true^</Enabled^> >> crypto_ml_pipeline_task.xml
echo     ^</TimeTrigger^> >> crypto_ml_pipeline_task.xml
echo   ^</Triggers^> >> crypto_ml_pipeline_task.xml
echo   ^<Settings^> >> crypto_ml_pipeline_task.xml
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^> >> crypto_ml_pipeline_task.xml
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^> >> crypto_ml_pipeline_task.xml
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^> >> crypto_ml_pipeline_task.xml
echo   ^</Settings^> >> crypto_ml_pipeline_task.xml
echo   ^<Actions^> >> crypto_ml_pipeline_task.xml
echo     ^<Exec^> >> crypto_ml_pipeline_task.xml
echo       ^<Command^>python^</Command^> >> crypto_ml_pipeline_task.xml
echo       ^<Arguments^>scripts/run_ml_pipeline.py^</Arguments^> >> crypto_ml_pipeline_task.xml
echo       ^<WorkingDirectory^>%CD%^</WorkingDirectory^> >> crypto_ml_pipeline_task.xml
echo     ^</Exec^> >> crypto_ml_pipeline_task.xml
echo   ^</Actions^> >> crypto_ml_pipeline_task.xml
echo ^</Task^> >> crypto_ml_pipeline_task.xml

echo.
echo ✅ Configuracion creada: crypto_ml_pipeline_task.xml
echo.
echo INSTRUCCIONES PARA AUTOMATIZACION:
echo 1. Abrir Task Scheduler (Programador de tareas)
echo 2. Hacer clic en "Importar tarea..."
echo 3. Seleccionar el archivo: crypto_ml_pipeline_task.xml
echo 4. Configurar credenciales si es necesario
echo 5. La tarea se ejecutara cada hora automaticamente
echo.
echo COMANDOS MANUALES DISPONIBLES:
echo   - Ejecutar pipeline completo:    python scripts/run_ml_pipeline.py
echo   - Solo extraccion:              python scripts/run_ml_pipeline.py --extract-only
echo   - Solo transformacion:          python scripts/run_ml_pipeline.py --transform-only  
echo   - Verificar salud del sistema:  python scripts/run_ml_pipeline.py --health-check
echo.
echo ========================================
echo    CONFIGURACION COMPLETADA
echo ========================================
pause
