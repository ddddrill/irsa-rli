$ErrorActionPreference = "Stop"
$env:PYINSTALLER_ISOLATED_BUILD = "0"

$venv = Join-Path $PSScriptRoot ".build_venv"
$python = Join-Path $venv "Scripts\python.exe"
$spec = Join-Path $PSScriptRoot "IRSA.spec"

# Create venv if missing
if (-not (Test-Path $python)) {
    Write-Host "Creating venv..."
    python -m venv $venv
    & $python -m pip install numpy scipy matplotlib PyQt5 opencv-python pyqtgraph Pillow "pyinstaller==5.13.2" "setuptools<71"
}

# Clean
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

Write-Host "Building IRSA..."
& $python -m PyInstaller $spec --noconfirm

Write-Host "`n=== Done: dist\IRSA\IRSA.exe ==="
