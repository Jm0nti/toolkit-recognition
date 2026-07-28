@echo off
REM Iniciador rapido del Detector de herramientas (Windows).
REM Doble clic en el Explorador, o desde cmd:  iniciar.bat
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No encuentro el entorno .venv.
  echo Crealo con:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

REM busca el primer puerto libre desde el 8000
set "PORT="
for %%p in (8000 8001 8002 8003 8004) do (
  if not defined PORT (
    netstat -ano | findstr ":%%p " | findstr LISTENING >nul || set "PORT=%%p"
  )
)
if not defined PORT (
  echo No hay puertos libres entre 8000 y 8004.
  pause
  exit /b 1
)

echo Detector de herramientas  ->  http://localhost:!PORT!
REM abre el navegador cuando el server responda (sondea el puerto)
start "" cmd /c "for /l %%i in (1,1,30) do (curl -s -o nul http://localhost:!PORT!/ && (start http://localhost:!PORT! ^& exit) || timeout /t 1 >nul)"

.venv\Scripts\python -m uvicorn scripts.serve_app:app --host 127.0.0.1 --port !PORT!
