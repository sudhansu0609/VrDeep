# VrDeep's environment, (re)built in place. Run by "VrDeep Setup.exe", or by hand:
#   powershell -ExecutionPolicy Bypass -File installer\setup_env.ps1
# Every step skips what is already in place, so re-running it repairs a half-finished or
# cleaned-up install (e.g. after a disk cleaner removed facefusion\venv).
. (Join-Path $PSScriptRoot "setup_common.ps1")
$root = $script:AppRoot
$facefusionDir = Join-Path $root "facefusion"

# setup_common.ps1 sets $ErrorActionPreference = "Stop". A `python -c` call
# whose import fails writes a traceback to stderr, and even redirected with
# `2>$null` that still raises a terminating NativeCommandError under "Stop"
# in Windows PowerShell 5.1 (PowerShell/PowerShell#4002) -- Invoke-Native
# already works around this by lowering it to "Continue" for the call; these
# two helpers do the same for the direct `python -c` probes below, without
# touching the shared template.
function Run-PyCheck([string]$Python, [string]$Code) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & $Python -c $Code 2>$null
    $ErrorActionPreference = $old
    return $out
}
function Safe-TestPyImport([string]$Python, [string]$Module) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $ok = Test-PyImport $Python $Module
    $ErrorActionPreference = $old
    return $ok
}

Set-Steps 8

Step "Checking the tools VrDeep needs (Python 3.13, ffmpeg)"
# FaceFusion's own requirements.txt pins scipy==1.18.0, which has no wheel below
# Python 3.12 (confirmed live: install.py's pip install fails to resolve it on
# 3.11). 3.12 is not on this machine's `py` launcher, so 3.13 is the closest
# version available with wheels for every pinned package (scipy, onnx, gradio,
# onnxruntime-gpu, opencv-python-headless) and for CUDA PyTorch (cu128).
$null = Find-Python "3.13"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-SetupLog "WARNING: ffmpeg is not on PATH. Install it (winget install Gyan.FFmpeg.Shared) before running a job."
}

Step "FaceFusion environment (facefusion\venv: Python 3.13)"
$py = Ensure-Venv (Join-Path $facefusionDir "venv") "3.13"

Step "FaceFusion's own CUDA install (onnxruntime-gpu via install.py cuda@12)"
# install.py resolves pip with shutil.which('pip'), so the venv's Scripts dir has to be
# first on PATH for that call or it installs into whichever pip happens to be on PATH.
$ffReady = (Safe-TestPyImport $py "onnxruntime") -and (Safe-TestPyImport $py "gradio") -and (Safe-TestPyImport $py "cv2")
if ($ffReady) {
    $ortVersion = Run-PyCheck $py "import onnxruntime; print(onnxruntime.__version__)"
    if ($LASTEXITCODE -ne 0 -or $ortVersion.Trim() -ne "1.24.4") { $ffReady = $false }
}
if ($ffReady) {
    Write-SetupLog "FaceFusion's CUDA environment present (onnxruntime-gpu 1.24.4, gradio, opencv)"
} else {
    $venvScripts = Split-Path $py -Parent
    $oldPath = $env:PATH
    $env:PATH = "$venvScripts;$oldPath"
    try {
        Invoke-Native "FaceFusion install.py cuda@12 --skip-conda" $py @("install.py", "cuda@12", "--skip-conda") $facefusionDir
    } finally {
        $env:PATH = $oldPath
    }
}

Step "CUDA runtime libraries for onnxruntime (cuDNN, cuBLAS, cuFFT, cuRAND)"
# onnxruntime-gpu ships no CUDA libraries of its own (PROGRESS.md, "SPEED & QUALITY FIX"):
# without these, onnxruntime_providers_cuda.dll fails to load and onnxruntime falls back to
# the CPU provider silently.
Pip-Install $py @("nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-runtime-cu12") "Installing CUDA runtime libraries for onnxruntime"

Step "Registering the CUDA DLL directories at interpreter start-up"
# onnxruntime only finds the pip-installed CUDA/cuDNN/cuBLAS DLLs if something calls
# onnxruntime.preload_dlls() -- FaceFusion never does. A sibling .pth file (the first word
# of a .pth line beginning "import " is executed by the site module at every interpreter
# start-up) runs this module so a plain `import onnxruntime` lands on CUDA, not the CPU.
$sitePackages = & $py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$dllModule = Join-Path $sitePackages "_vrdeep_cuda_dlls.py"
$dllPth = Join-Path $sitePackages "_vrdeep_cuda_dlls.pth"
if (-not (Test-Path $dllModule) -or -not (Test-Path $dllPth)) {
    $moduleBody = @'
"""Registers the CUDA/cuDNN/cuBLAS/cuFFT/cuRAND DLL directories installed via
pip's nvidia-*-cu12 wheels so onnxruntime's CUDAExecutionProvider can load
without PATH changes. onnxruntime-gpu ships no CUDA libraries of its own and
only finds these if something calls preload_dlls() -- FaceFusion never does,
so the sibling _vrdeep_cuda_dlls.pth runs this at every interpreter start-up.
"""
try:
    import onnxruntime
    onnxruntime.preload_dlls()
except Exception:
    pass
'@
    Set-Content -Path $dllModule -Value $moduleBody -Encoding utf8
    Set-Content -Path $dllPth -Value "import _vrdeep_cuda_dlls" -Encoding utf8
    Write-SetupLog "wrote $dllModule and $dllPth"
} else {
    Write-SetupLog "CUDA DLL preload hook present: $dllModule"
}

Step "CUDA PyTorch (cu128; the 5060 Ti is Blackwell/sm_120) for the Real-ESRGAN upscale path"
# Ensure-CudaTorch's own "already present?" probe is the same raw `python -c`
# pattern Test-PyImport uses (see the note above); protect it the same way.
# No -Packages override: a single-element list collapses to a plain string in
# Ensure-CudaTorch's own `$pkgs = if (...) {...} else { $Packages }` (a lone
# object from an if/else branch is unrolled, not kept as a 1-item array),
# which turns its later `+` into string concatenation and mangles the pip
# command line (seen live: "torch--index-url https://..." as one argument).
# The built-in default (torch, torchaudio) is two elements, so it is not hit.
$old = $ErrorActionPreference
$ErrorActionPreference = "Continue"
Ensure-CudaTorch $py
$ErrorActionPreference = $old

Step "VrDeep's own GUI dependencies (customtkinter, Pillow)"
Pip-Install $py @("customtkinter", "Pillow") "Installing the desktop UI dependencies"

Step "Checking the result"
$providers = Run-PyCheck $py "import onnxruntime as ort; p = ort.get_available_providers(); print('onnxruntime providers:', p); import sys; sys.exit(0 if 'CUDAExecutionProvider' in p else 1)"
Write-SetupLog $providers
if ($LASTEXITCODE -ne 0) { Fail "onnxruntime does not list CUDAExecutionProvider. See $script:SetupLog" }
$scriptsDir = Join-Path $root "scripts"
$appImports = Run-PyCheck $py "import sys; sys.path.insert(0, r'$scriptsDir'); import customtkinter, tkinter; from PIL import Image; import engine, guardrails, vr_metadata, upscale_vr, clip_selector, character_picker; print('app.py imports OK')"
Write-SetupLog $appImports
if ($LASTEXITCODE -ne 0) { Fail "scripts\app.py's own imports do not resolve. See $script:SetupLog" }
$torchCheck = Run-PyCheck $py "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| available', torch.cuda.is_available())"
Write-SetupLog $torchCheck
if ($LASTEXITCODE -ne 0) { Fail "torch does not import" }
Done "VrDeep's environment is ready. Models go under VrDeep\models\ as before."
