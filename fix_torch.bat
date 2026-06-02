@echo off
chcp 65001 >nul 2>nul
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================
echo   Fix Environment (CUDA torch + web deps)
echo ============================================
echo.
echo Make sure the web server is STOPPED first!
echo.

echo [1/2] Reinstalling CUDA torch...
uv sync --extra cuda

echo.
echo [2/2] Reinstalling web UI dependencies...
uv pip install fastapi uvicorn aiofiles python-multipart

echo.
echo Done! You can now run start_webui.bat
echo.
pause
