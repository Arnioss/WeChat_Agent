@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0Python3.12\python.exe"

if not exist "%PYTHON%" (
  echo [check_env] Python not found: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" -m pip check
if errorlevel 1 goto failed

"%PYTHON%" check_env.py
if errorlevel 1 goto failed

pause
exit /b 0

:failed
pause
exit /b 1
