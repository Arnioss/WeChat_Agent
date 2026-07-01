@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PYTHON=%~dp0Python3.12\python.exe"

if not exist "%PYTHON%" (
  echo [Streamlit] 未找到便携 Python: %PYTHON%
  pause
  exit /b 1
)

echo [Streamlit] 检查 8501 端口是否有残留进程...
set "FOUND_OLD=0"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
  set "FOUND_OLD=1"
  echo [Streamlit] 结束残留进程 PID=%%p
  taskkill /F /PID %%p >nul 2>&1
)
if "!FOUND_OLD!"=="1" timeout /t 2 /nobreak >nul

echo [Streamlit] 正在启动...
echo [Streamlit] 本机:   http://127.0.0.1:8501
echo [Streamlit] 局域网: http://你的局域网IP:8501
echo [Streamlit] 运行 ipconfig 查看 IPv4，例如 http://10.x.x.x:8501
echo [Streamlit] 首次加载 Agent 可能需要 10-20 秒。
"%PYTHON%" -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
pause
