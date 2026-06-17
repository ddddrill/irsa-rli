@echo off
echo === Building IRSA ===

if not exist .build_venv (
    echo Creating venv...
    python -m venv .build_venv
    .build_venv\Scripts\pip.exe install "numpy<2" "scipy<1.12" "matplotlib<3.9" PyQt5 opencv-python pyqtgraph Pillow "pyinstaller==5.13.2" "setuptools<71"
)

if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo Building...
.build_venv\Scripts\python.exe -m PyInstaller IRSA.spec --noconfirm

echo === Done: dist\IRSA\IRSA.exe ===
pause
