$ErrorActionPreference = "Stop"

$PROJECT = "B:\youtubeProjects\Buzzcaf Media\VrDeep"
$PYTHON = "$PROJECT\facefusion\venv\Scripts\python.exe"
$PIPELINE = "$PROJECT\scripts\run_pipeline.py"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VR DEEPFAKE PIPELINE - QUICK START" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$source = Read-Host "Source face image path (or press Enter to skip)"
$target = Read-Host "Target VR video path (or press Enter to skip)"

if (-not $source -or -not $target) {
    Write-Host "`nUsage: Place source face in input\source\ and target video in input\target\" -ForegroundColor Yellow
    Write-Host "Then run: $PIPELINE <source_face> <target_video>" -ForegroundColor Yellow
    exit
}

$layout = Read-Host "Stereo layout (sbs/tb) [sbs]"
if (-not $layout) { $layout = "sbs" }

$outputName = "deepfake_$(Get-Date -Format 'yyyyMMdd_HHmmss').mp4"
$outputPath = "$PROJECT\output\final\$outputName"

Write-Host "`nStarting pipeline..." -ForegroundColor Green
Write-Host "  Source: $source" -ForegroundColor White
Write-Host "  Target: $target" -ForegroundColor White
Write-Host "  Layout: $layout" -ForegroundColor White
Write-Host "  Output: $outputPath" -ForegroundColor White
Write-Host ""

& $PYTHON $PIPELINE `
    $source `
    $target `
    -o $outputPath `
    -l $layout `
    -c "$PROJECT\scripts\pipeline_config.json"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDone! Output saved to:" -ForegroundColor Green
    Write-Host "  $outputPath" -ForegroundColor Green
} else {
    Write-Host "`nPipeline failed!" -ForegroundColor Red
}
