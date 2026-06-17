@echo off
echo === Building IRSA ===

echo [1/2] Installing PyInstaller...
pip install pyinstaller

echo [2/2] Building IRSA...
set PYINSTALLER_ISOLATED_BUILD=0
pyinstaller IRSA.spec --noconfirm

echo.
echo === Build complete ===
echo Output: dist\IRSA\IRSA.exe
pause
