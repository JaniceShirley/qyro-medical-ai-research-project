@echo off
rem Start the QYRO demo on Windows at http://localhost:8000
rem API keys (optional) go in qyro_app\.env, see qyro_app\.env.example
cd /d "%~dp0.."
if not exist qyro_app\.env type nul > qyro_app\.env
set PYTHONWARNINGS=ignore
if "%PORT%"=="" set PORT=8000
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
echo Starting QYRO demo with %PY% on http://localhost:%PORT% ...
"%PY%" -m uvicorn qyro_app.main:app --host 127.0.0.1 --port %PORT% --env-file qyro_app\.env
pause
