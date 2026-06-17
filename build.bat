@echo off
echo === Building IRSA ===

if not exist .build_venv (
    echo Creating venv...
    python -m venv .build_venv
    .build_venv\Scripts\pip.exe install numpy scipy matplotlib PyQt5 opencv-python pyqtgraph Pillow "pyinstaller==5.13.2" "setuptools<71"
)

if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo Building...
.build_venv\Scripts\python.exe -m PyInstaller IRSA.spec --noconfirm

echo === Done: dist\IRSA\IRSA.exe ===
pause
