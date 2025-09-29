@echo off
echo ========================================
echo   CONFIGURACION DE GITHUB REMOTE
echo ========================================

echo.
echo IMPORTANTE: Reemplaza TU_USUARIO con tu usuario real de GitHub
echo.

echo Comando para conectar con GitHub:
echo git remote add origin https://github.com/TU_USUARIO/crypto-etl-pipeline.git
echo.

echo Comando para subir el codigo:
echo git branch -M main
echo git push -u origin main
echo.

echo Una vez ejecutados estos comandos, continua con la configuracion de secretos.
echo.
pause
