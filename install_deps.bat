@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0Python3.12\python.exe"

if not exist "%PYTHON%" (
  echo [install_deps] Python not found: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" -m pip install -U -r requirements.txt
if errorlevel 1 goto failed

"%PYTHON%" -m pip check
if errorlevel 1 goto failed

pause
exit /b 0

:failed
pause
exit /b 1
