# VrDeep Desktop App Launcher
# Run this from PowerShell to launch the desktop app.
#
# Starts pythonw.exe, not python.exe, so the app does not hold a console open
# behind itself - the window you see is the app and nothing else. This prompt
# comes straight back; anything the app prints goes to output\app.log.
# Use launch_debug.bat instead when you need to watch that output live.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectDir = $scriptDir
$pythonPath = Join-Path $projectDir "facefusion\venv\Scripts\pythonw.exe"
$appPath = Join-Path $projectDir "scripts\app.py"

if (-Not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Python not found at: $pythonPath" -ForegroundColor Red
    Write-Host "Make sure FaceFusion is set up in the 'facefusion' folder." -ForegroundColor Yellow
    exit 1
}

if (-Not (Test-Path $appPath)) {
    Write-Host "ERROR: app.py not found at: $appPath" -ForegroundColor Red
    exit 1
}

# The CUDA and TensorRT DLL directories. Without them the GPU providers fall
# back to CPU, which is the difference between minutes and hours.
$torchLib = Join-Path $projectDir "facefusion\venv\Lib\site-packages\torch\lib"
$trtLib = Join-Path $projectDir "facefusion\venv\Lib\site-packages\tensorrt_libs"
if (Test-Path $torchLib) { $env:PATH = "$torchLib;$env:PATH" }
if (Test-Path $trtLib) { $env:PATH = "$trtLib;$env:PATH" }

Write-Host "Launching VrDeep desktop app..." -ForegroundColor Green
Start-Process -FilePath $pythonPath -ArgumentList $appPath
