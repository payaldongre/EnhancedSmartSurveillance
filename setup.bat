@echo off
REM ===========================================================================
REM One-time setup for EnhancedSmartSurveillance (Windows, conda).
REM Creates .\env with Python 3.10 and installs every requirement, including
REM the CPU build of PyTorch. After this, starting the app is just:
REM     conda activate <this folder>\env
REM     python app.py
REM ===========================================================================
setlocal
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found on PATH.
    echo Install Miniconda/Anaconda, or set up a venv manually:
    echo     python -m venv env
    echo     env\Scripts\activate
    echo     pip install -r requirements.txt
    exit /b 1
)

if not exist "env" (
    echo Creating conda environment ".\env" with Python 3.10 ...
    call conda create --prefix ".\env" python=3.10 -y
    if errorlevel 1 exit /b 1
)

call conda activate ".\env"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Start the app with:
echo     conda activate "%~dp0env"
echo     python app.py
