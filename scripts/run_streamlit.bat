@echo off
cd /d "%~dp0.."
"%~dp0..\.venv\Scripts\python.exe" -m streamlit run frontend\main.py --server.headless true
pause
