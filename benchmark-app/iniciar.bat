@echo off
rem Abre o app de benchmark do Juiz de Bocha em http://localhost:8501
cd /d "%~dp0"

set "UV=uv"
where uv >nul 2>nul || set "UV=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
if not "%UV%"=="uv" if not exist "%UV%" (
    echo Nao encontrei o uv. Instale com:  winget install --id astral-sh.uv -e
    pause
    exit /b 1
)

"%UV%" run streamlit run app.py --server.port 8501 --server.headless false
if errorlevel 1 pause
