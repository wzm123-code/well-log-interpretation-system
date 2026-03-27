@echo off
cd /d "%~dp0..\backend"
"%~dp0..\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
