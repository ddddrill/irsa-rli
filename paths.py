import sys
import os


def resource_path(relative_path):
    """Абсолютный путь к читаемому ресурсу (matrices/, sat_info_p/, assets/).

    Внутри PyInstaller .exe берёт путь из sys._MEIPASS (временная папка).
    При запуске из исходников — относительно текущего файла.
    """
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def app_dir():
    """Каталог для записи данных (config.json, logs/, radioimage/).

    Рядом с .exe при сборке, рядом с main.py при запуске из исходников.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))
