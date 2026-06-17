@echo off
echo === Building IRSA ===

set PYINSTALLER_ISOLATED_BUILD=0
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

.build_venv\Scripts\python.exe -m PyInstaller IRSA.spec --noconfirm

echo === Done: dist\IRSA\IRSA.exe ===
pause
