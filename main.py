import os
import sys
import logging
import logging.handlers

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from paths import resource_path, app_dir
from ui.main_window import MainWindow

LOG_DIR = os.path.join(app_dir(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "isar.log"),
    maxBytes=2 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("assets/icon.ico")))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    logger.info("Приложение IRSA запущено")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
