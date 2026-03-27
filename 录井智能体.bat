@echo off
REM Copy this file to Desktop. Edit APP_ROOT if your project is not here.
set "APP_ROOT=D:\Python\app"

cd /d "%APP_ROOT%" 2>nul
if not exist "%APP_ROOT%\scripts\start_all.bat" (
  echo [ERROR] Cannot find: %APP_ROOT%\scripts\start_all.bat
  echo Edit DesktopLaunch.bat and set APP_ROOT= to your project folder.
  pause
  exit /b 1
)

call "%APP_ROOT%\scripts\start_all.bat"
if errorlevel 1 pause
