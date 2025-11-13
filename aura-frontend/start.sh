@echo off
REM Aura Frontend Startup Script (Windows)

echo 🎙️ Starting Aura Frontend...
echo.

REM Check if backend is running
echo Checking backend connection...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running
) else (
    echo ⚠️  Backend is not running on http://localhost:8000
    echo    Please start the backend first:
    echo    cd ..\aura-backend ^&^& python main.py
    echo.
    echo    Or start with Docker:
    echo    cd .. ^&^& docker-compose up
    echo.
    set /p continue="Continue anyway? (y/n) "
    if /i not "%continue%"=="y" exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo 🚀 Starting Streamlit frontend...
echo.
echo Access the UI at: http://localhost:8501
echo Backend API docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

REM Start Streamlit
streamlit run streamlit_app.py
