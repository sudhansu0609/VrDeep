@echo off
setlocal enabledelayedexpansion

REM VrDeep - launch the app with a console attached.
REM
REM This is the old launcher, kept for when the app will not start or exits on
REM its own: python.exe rather than pythonw.exe, so tracebacks and library
REM warnings land in this window instead of only in output\app.log, and the
REM window stays open afterwards to show them.
REM
REM For normal use, launch_app.vbs (or launch_app.bat) shows the app alone.

echo ========================================
echo   VrDeep - debug launch (console shown)
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%facefusion\venv\Scripts\python.exe"
set "APP_PY=%SCRIPT_DIR%scripts\app.py"
set "TORCH_LIB=%SCRIPT_DIR%facefusion\venv\Lib\site-packages\torch\lib"
set "TRT_LIB=%SCRIPT_DIR%facefusion\venv\Lib\site-packages\tensorrt_libs"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at: %PYTHON_EXE%
    echo Make sure FaceFusion is set up in the 'facefusion' folder.
    pause
    exit /b 1
)

if not exist "%APP_PY%" (
    echo ERROR: app.py not found at: %APP_PY%
    pause
    exit /b 1
)

REM Add CUDA and TensorRT libraries to PATH for GPU acceleration
if exist "%TORCH_LIB%" (
    set "PATH=%TORCH_LIB%;%PATH%"
    echo [OK] CUDA libraries loaded from PyTorch
)
if exist "%TRT_LIB%" (
    set "PATH=%TRT_LIB%;%PATH%"
    echo [OK] TensorRT libraries loaded
)

echo.
echo Launching VrDeep desktop app...
"%PYTHON_EXE%" "%APP_PY%"

echo.
echo App exited with code %ERRORLEVEL%.
pause
