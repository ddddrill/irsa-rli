"""Build script for IRSA.
Sets PYINSTALLER_ISOLATED_BUILD before any PyInstaller import.
"""
import os
import subprocess
import sys
import shutil

os.environ["PYINSTALLER_ISOLATED_BUILD"] = "0"

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, ".build_venv")
DIST_DIR = os.path.join(ROOT, "dist")
BUILD_DIR = os.path.join(ROOT, "build")

def main():
    print("=== Building IRSA ===")

    # Clean previous builds
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(d):
            print(f"Cleaning {d}...")
            shutil.rmtree(d)

    # Create clean venv if needed
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        print("[1/3] Creating clean venv...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

        pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        print("[2/3] Installing dependencies...")
        subprocess.check_call([
            pip, "install",
            "numpy", "scipy", "matplotlib", "PyQt5",
            "opencv-python", "pyqtgraph", "Pillow", "pyinstaller",
        ])
    else:
        print("[1/3] Venv exists, skipping creation")
        print("[2/3] Dependencies already installed")

    print("[3/3] Building IRSA...")
    subprocess.check_call([
        venv_python, "-m", "PyInstaller",
        os.path.join(ROOT, "IRSA.spec"),
        "--noconfirm",
    ])

    print()
    print("=== Build complete ===")
    print(f"Output: {DIST_DIR}\\IRSA\\IRSA.exe")

if __name__ == "__main__":
    main()
