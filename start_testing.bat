@echo off
REM Robotics Model Optimization Platform - Testing Startup Script (Windows)
REM This script helps you quickly start the API server and run tests

echo ==========================================
echo 🤖 Robotics Model Optimization Platform
echo    Testing Startup Script (Windows)
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo ℹ️  Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dependencies not installed. Installing...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies already installed
)

REM Show menu
echo.
echo ==========================================
echo What would you like to do?
echo ==========================================
echo 1) Start API Server
echo 2) Run Automated Test Script
echo 3) Start Frontend
echo 4) Run Integration Tests
echo 5) View API Documentation
echo 6) Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto run_test
if "%choice%"=="3" goto start_frontend
if "%choice%"=="4" goto run_integration
if "%choice%"=="5" goto view_docs
if "%choice%"=="6" goto exit_script
goto invalid_choice

:start_api
echo ℹ️  Starting API Server...
echo.
echo ✅ API Server will be available at:
echo    - API: http://localhost:8000
echo    - Docs: http://localhost:8000/docs
echo    - Health: http://localhost:8000/health
echo.
echo ℹ️  Press Ctrl+C to stop the server
echo.
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
goto end

:run_test
echo ℹ️  Running automated test script...
echo.
echo ⚠️  Make sure API server is running in another terminal!
echo.
pause
python test_real_model.py
goto end

:start_frontend
echo ℹ️  Starting Frontend...
echo.
if not exist "frontend\node_modules" (
    echo ⚠️  Node modules not installed. Installing...
    cd frontend
    call npm install
    cd ..
    echo ✅ Node modules installed
)
echo.
echo ✅ Frontend will be available at: http://localhost:3000
echo.
echo ℹ️  Press Ctrl+C to stop the frontend
echo.
cd frontend
call npm start
cd ..
goto end

:run_integration
echo ℹ️  Running integration tests...
echo.
pytest tests\integration\test_real_optimization_workflow.py -v -s
goto end

:view_docs
echo ℹ️  Opening API documentation...
echo.
echo ✅ API Documentation URLs:
echo    - Swagger UI: http://localhost:8000/docs
echo    - ReDoc: http://localhost:8000/redoc
echo.
echo ⚠️  Make sure API server is running!
echo.
start http://localhost:8000/docs
goto end

:invalid_choice
echo ❌ Invalid choice
pause
exit /b 1

:exit_script
echo ℹ️  Exiting...
exit /b 0

:end
echo.
echo ✅ Done!
pause
