"""Build script for IRSA.
Uses .build_venv with PyInstaller (PYINSTALLER_ISOLATED_BUILD=0).
"""
import os
import subprocess
import shutil

os.environ["PYINSTALLER_ISOLATED_BUILD"] = "0"

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ROOT, ".build_venv", "Scripts", "python.exe")
DIST_DIR = os.path.join(ROOT, "dist")
BUILD_DIR = os.path.join(ROOT, "build")


def main():
    print("=== Building IRSA ===")

    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print("Building...")
    subprocess.check_call([VENV_PYTHON, "-m", "PyInstaller", os.path.join(ROOT, "IRSA.spec"), "--noconfirm"])

    print(f"\n=== Done: {DIST_DIR}\\IRSA\\IRSA.exe ===")


if __name__ == "__main__":
    main()
