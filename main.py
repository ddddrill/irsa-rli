import sys
import logging

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    logger.info("Приложение IRSA запущено")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
