@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0Python3.12\python.exe"

if not exist "%PYTHON%" (
  echo [WeCom] Python not found: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" wecom_ws_server.py
pause
