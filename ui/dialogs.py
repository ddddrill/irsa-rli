import math
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QDialog, QVBoxLayout

import target
from config import AppConfig
from filenames import get_matrix_filename
from processing import DataProcessor


class GraphWindow(QDialog):
    """Окно вывода отдельного РЛИ с интерактивным графиком."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Радиолокационное изображение")
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self._plot()

    def _plot(self):
        cfg = AppConfig.load()
        filename = get_matrix_filename(cfg.satellite)

        D = DataProcessor(
            filename=filename,
            method=cfg.method,
            f_c=cfg.center_frequency,
            Xmax=cfg.survey_length,
            Ymax=cfg.survey_width,
            spectr_w=cfg.spectrum_width,
            ang=cfg.beam_width,
            range_m=cfg.range_m,
        )

        h = 3e8 / (2 * cfg.center_frequency * 1e9)
        T = target.Target(filename, True)
        raw_tensor = T.targets_matrix()
        intens, x, y = raw_tensor[0]
        x_1 = x - math.floor((max(x) - min(x)) / 2)
        y_1 = y - math.floor((max(y) - min(y)) / 2)

        ri = D.radioimage_single(intens, x_1, y_1)
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if cfg.method == "стандартный":
            ax.contourf(
                abs(np.rot90(ri[3], 2)),
                extent=[ri[4][0], ri[4][-1], ri[5][0] * h, ri[5][-1] * h],
                cmap="plasma",
            )
        elif cfg.method == "с полярным переформатированием":
            ax.contourf(
                ri[7], ri[6],
                abs(np.rot90(ri[3], 2) / (ri[4] * ri[5])),
                cmap="plasma",
            )

        ax.set_xlabel("Ширина КА, м")
        ax.set_ylabel("Длина КА, м")
        self.canvas.draw()
