# VrDeep - VR Face-Swap Pipeline

A desktop app (customtkinter) that runs FaceFusion face-swapping over stereoscopic VR video:
it splits each eye, swaps faces through FaceFusion (CUDA), optionally upscales
(Real-ESRGAN or FaceFusion's own ONNX frame enhancer) and interpolates frames (RIFE), then
merges the eyes back and writes VR metadata into the output. `scripts/guardrails.py` caps
memory/CPU per run so a job can't take the machine down.

## Install

Run **`installer\VrDeep Setup.exe`**. It installs VrDeep in place (in this folder) and:

1. builds the environment (`installer\setup_env.ps1`): `facefusion\venv` (Python 3.13),
   FaceFusion's own CUDA onnxruntime (`facefusion\install.py cuda@12`), the CUDA runtime
   libraries onnxruntime-gpu needs (cuDNN, cuBLAS, cuFFT, cuRAND), CUDA PyTorch (cu128, for
   the Blackwell 5060 Ti) for the Real-ESRGAN upscale path, and VrDeep's own GUI dependencies
   (customtkinter, Pillow);
2. makes the Desktop and Start Menu (Buzzcaf Media) shortcuts to `launch_app.vbs`;
3. registers VrDeep in Settings > Apps and writes `installer/installed.json`, which keeps
   SmartCleaner from offering `facefusion\venv` for cleanup.

Needs Python 3.13 (`py -3.13`) and ffmpeg on PATH. Re-run Setup to repair an install (e.g.
after a disk cleaner removes `facefusion\venv`): each step skips what is already there.
`installer\Uninstall VrDeep.exe` removes the shortcuts and the Apps entry and keeps the
folder, its models and its environment. Without the wizard:

```powershell
powershell -ExecutionPolicy Bypass -File installer\setup_env.ps1
```

## Run

Use the Desktop/Start Menu shortcut, or double-click `launch_app.vbs` directly (no console
window; anything the app prints goes to `output\app.log`). `launch_debug.bat` runs it with a
console attached instead, for when the app won't start or exits on its own.

## Models

`models\rife\` holds the RIFE frame-interpolation weights (downloaded on demand by
`scripts\rife_worker.py --install-model`). The optional Real-ESRGAN upscale path reads its
weights from `facefusion\weights\realesr-general-x4v3.pth` (drop the file there, or leave
`upscale_method` on `ffmpeg` / FaceFusion's own `real_esrgan_x2_fp16` frame enhancer instead).
FaceFusion's own face-swap/enhancer models download under `facefusion\.assets\` on first use.
