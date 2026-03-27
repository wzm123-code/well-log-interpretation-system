@echo off
REM Avoid chcp UTF-8 here: can break parsing when double-clicked from some locales.
title Smart Well Log Launcher
cd /d "%~dp0.."
set "APP_ROOT=%CD%"

if not exist "%APP_ROOT%\frontend\main.py" goto err_no_project
if not exist "%APP_ROOT%\.venv\Scripts\python.exe" goto err_no_venv

echo.
echo Project: %APP_ROOT%
echo.

echo [1/2] Starting backend API (8000)...
start "WellLog-Backend-8000" cmd /k "%APP_ROOT%\scripts\run_backend.bat"

REM Delay ~3s without timeout (timeout can fail when stdin is not a console)
ping 127.0.0.1 -n 4 >nul

echo [2/2] Starting Streamlit (8501)...
start "WellLog-Streamlit-8501" cmd /k "%APP_ROOT%\scripts\run_streamlit.bat"

ping 127.0.0.1 -n 6 >nul
start http://127.0.0.1:8501

echo.
echo Browser should open. Closing this window does NOT stop services.
echo Close the two titled console windows to stop backend and Streamlit.
echo.
pause
goto :eof

:err_no_project
echo [ERROR] frontend\main.py not found under:
echo %APP_ROOT%
exit /b 1

:err_no_venv
echo [ERROR] Virtual env not found:
echo %APP_ROOT%\.venv\Scripts\python.exe
echo Create venv:  python -m venv .venv
echo Install deps:  .venv\Scripts\pip install -r requirements-venv.txt
exit /b 1
