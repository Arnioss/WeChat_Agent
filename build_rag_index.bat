@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0Python3.12\python.exe"

if not exist "%PYTHON%" (
  echo [RAG] Python not found: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" build_rag_index.py
pause
