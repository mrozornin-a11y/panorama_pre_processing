@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения .venv ...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 goto fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto fail
) else (
    ".venv\Scripts\python.exe" -c "import PySide6, PIL, numpy, cv2" 2>nul || ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
exit /b 0

:fail
echo.
echo Не удалось подготовить окружение. Убедитесь, что установлен Python 3.13 (python.org) и есть доступ в интернет.
rmdir /s /q .venv 2>nul
pause
exit /b 1
