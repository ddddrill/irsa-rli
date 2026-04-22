import os
import logging
import math

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QProgressBar, QComboBox,
    QLineEdit, QPushButton, QGroupBox, QVBoxLayout,
    QHBoxLayout, QSplitter, QScrollArea, QWidget,
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPixmap

from config import AppConfig
from filenames import get_matrix_filename
from processing import DataProcessor
from ui.dialogs import GraphWindow
from ui.styles import APP_STYLE

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SATELLITE_DIR = os.path.join(BASE_DIR, "sat_info_p")
SPEED_OF_LIGHT = 3e8
NUM_IMAGES = 16


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cfg = AppConfig.load()
        self.processor = None
        self._frame_index = 0
        self._raw_tensor = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("IRSA")
        self.resize(1280, 720)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(APP_STYLE)
        pg.setConfigOptions(imageAxisOrder="row-major")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        # вертикальный сплиттер: верх / низ
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(3)
        main_splitter.addWidget(self._build_upper_panel())
        main_splitter.addWidget(self._build_lower_panel())
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([300, 420])

        root_layout.addWidget(main_splitter)
        self.statusBar().showMessage("Готово к работе")

    # ================================================================
    #  Верхняя панель: спутник (лево) | параметры + кнопки (право)
    # ================================================================

    def _build_upper_panel(self):
        # горизонтальный сплиттер: спутник | параметры
        hsplit = QSplitter(Qt.Horizontal)
        hsplit.setHandleWidth(2)
        hsplit.addWidget(self._build_satellite_box())
        hsplit.addWidget(self._build_params_and_buttons())
        hsplit.setStretchFactor(0, 1)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([450, 450])
        return hsplit

    # ---------- спутник (лево) ----------

    def _build_satellite_box(self):
        grp = QGroupBox("Исследуемый спутник")
        layout = QHBoxLayout(grp)
        layout.setSpacing(8)

        # картинка (слева)
        self.image_label_sat = QLabel("Нажмите «Показать КА»")
        self.image_label_sat.setAlignment(Qt.AlignCenter)
        self.image_label_sat.setFixedWidth(140)
        self.image_label_sat.setMinimumHeight(100)
        self.image_label_sat.setStyleSheet(
            "background-color: #fafafa; border: 1px dashed #bdc3c7;"
            "border-radius: 6px; padding: 4px;"
        )
        layout.addWidget(self.image_label_sat, stretch=0)

        # описание (справа, занимает основное место)
        self.sat_info = QLabel("")
        self.sat_info.setObjectName("sat_info")
        self.sat_info.setWordWrap(True)
        self.sat_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidget(self.sat_info)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll, stretch=1)

        return grp

    # ---------- параметры + кнопки (право) ----------

    def _build_params_and_buttons(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # столбик 1: параметры цели
        col1 = self._build_target_params_column()
        layout.addLayout(col1, stretch=1)

        # столбик 2: параметры радара
        col2 = self._build_radar_params_column()
        layout.addLayout(col2, stretch=1)

        # столбик 3: кнопки + прогресс
        col3 = self._build_buttons_column()
        layout.addLayout(col3, stretch=0)

        return panel

    def _build_target_params_column(self):
        col = QVBoxLayout()
        col.setSpacing(4)
        self.cb_satellite = self._make_combo(
            col, "Спутник",
            ["cloudSAT", "calipso", "ICEsat2", "LRO", "solarB"],
            self.cfg.satellite,
        )
        self.cb_method = self._make_combo(
            col, "Метод обработки",
            ["стандартный", "с полярным переформатированием"],
            self.cfg.method,
        )
        self.le_range = self._make_input(col, "Дальность, км", self.cfg.range_km)
        return col

    def _build_radar_params_column(self):
        col = QVBoxLayout()
        col.setSpacing(4)
        self.le_frequency = self._make_input(col, "Частота, ГГц", self.cfg.center_frequency)
        self.le_spectrum = self._make_input(col, "Спектр, ГГц", self.cfg.spectrum_width)
        self.le_nifft = self._make_input(col, "Размер FFT", self.cfg.nifft_size)
        self.le_beam = self._make_input(col, "Ширина ДН,град", self.cfg.beam_width)
        self.le_survey_length = self._make_input(col, "Длина обзора, м", self.cfg.survey_length)
        self.le_survey_width = self._make_input(col, "Ширина обзора, м", self.cfg.survey_width)
        return col

    def _build_buttons_column(self):
        col = QVBoxLayout()
        col.setSpacing(6)

        self.btn_show_sat = self._make_button(col, "Показать КА", self._show_satellite)
        self.btn_single = self._make_button(col, "Расчёт одиночного РЛИ", self._calc_single_rli)
        self.btn_stream = self._make_button(col, "Расчёт потока РЛИ", self._calc_stream, style_class="btn_action")
        self.btn_show_results = self._make_button(col, "Вывести результаты", self._show_results)

        col.addSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        col.addWidget(self.progress_bar)

        self.info_label = QLabel("")
        self.info_label.setObjectName("status_info")
        col.addWidget(self.info_label)

        col.addStretch()
        return col

    # ================================================================
    #  Нижняя панель: РЛИ
    # ================================================================

    def _build_lower_panel(self):
        grp = QGroupBox("Результаты обработки")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        # --- pyqtgraph: поле обратного рассеяния ---
        self.pg_field = pg.PlotWidget(background="#fafafa")
        self.pg_field.setLabel("left", "Частота, Гц")
        self.pg_field.setLabel("bottom", "Ширина ДН, радиан")
        self.pg_field_view = self.pg_field.getPlotItem().getViewBox()
        self.pg_field_img = pg.ImageItem()
        self.pg_field_view.addItem(self.pg_field_img)

        # --- pyqtgraph: РЛИ ---
        self.pg_result = pg.PlotWidget(background="#fafafa")
        self.pg_result.setLabel("left", "Y, м")
        self.pg_result.setLabel("bottom", "X, м")
        self.pg_result_view = self.pg_result.getPlotItem().getViewBox()
        self.pg_result_img = pg.ImageItem()
        self.pg_result_view.addItem(self.pg_result_img)

        imgs_layout = QHBoxLayout()
        imgs_layout.addWidget(self.pg_field, stretch=1)
        imgs_layout.addWidget(self.pg_result, stretch=1)

        field_lbl = QLabel("Поле обратного рассеяния")
        field_lbl.setObjectName("status_info")
        field_lbl.setAlignment(Qt.AlignCenter)
        result_lbl = QLabel("РЛИ")
        result_lbl.setObjectName("status_info")
        result_lbl.setAlignment(Qt.AlignCenter)

        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(field_lbl)
        lbl_layout.addWidget(result_lbl)

        layout.addLayout(lbl_layout)
        layout.addLayout(imgs_layout)

        return grp

    # ================================================================
    #  Фабрики виджетов
    # ================================================================

    def _make_combo(self, layout, label_text, items, current):
        lbl = QLabel(label_text)
        lbl.setObjectName("status_info")
        layout.addWidget(lbl)
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(self._on_config_changed)
        layout.addWidget(combo)
        layout.addSpacing(4)
        return combo

    def _make_input(self, layout, label_text, value):
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel(label_text)
        lbl.setWordWrap(True)
        row.addWidget(lbl, stretch=1)
        le = QLineEdit(str(value))
        le.setMaximumWidth(80)
        le.textChanged.connect(self._on_config_changed)
        row.addWidget(le)
        layout.addLayout(row)
        layout.addSpacing(2)
        return le

    def _make_button(self, layout, text, slot, style_class=None):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        if style_class:
            btn.setObjectName(style_class)
        layout.addWidget(btn)
        return btn

    # ================================================================
    #  Конфигурация
    # ================================================================

    def _safe_float(self, widget, fallback):
        try:
            return float(widget.text())
        except ValueError:
            return fallback

    def _safe_int(self, widget, fallback):
        try:
            return int(widget.text())
        except ValueError:
            return fallback

    def _on_config_changed(self):
        self.cfg.method = self.cb_method.currentText()
        self.cfg.satellite = self.cb_satellite.currentText()
        self.cfg.center_frequency = self._safe_float(self.le_frequency, self.cfg.center_frequency)
        self.cfg.spectrum_width = self._safe_float(self.le_spectrum, self.cfg.spectrum_width)
        self.cfg.nifft_size = self._safe_int(self.le_nifft, self.cfg.nifft_size)
        self.cfg.beam_width = self._safe_float(self.le_beam, self.cfg.beam_width)
        self.cfg.survey_length = self._safe_float(self.le_survey_length, self.cfg.survey_length)
        self.cfg.survey_width = self._safe_float(self.le_survey_width, self.cfg.survey_width)
        self.cfg.range_km = self._safe_float(self.le_range, self.cfg.range_km)
        self.cfg.save()

    # ================================================================
    #  Кнопка 1: Показать КА
    # ================================================================

    def _show_satellite(self):
        self._on_config_changed()
        sat_name = self.cfg.satellite
        self.statusBar().showMessage(f"Загрузка: {sat_name}")

        image_path = os.path.join(SATELLITE_DIR, f"{sat_name}.png")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.image_label_sat.setPixmap(
                pixmap.scaled(self.image_label_sat.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.image_label_sat.setStyleSheet(
                "background-color: #fafafa; border: 1px solid #dcdde1;"
                "border-radius: 6px; padding: 4px;"
            )
        else:
            self.image_label_sat.setText(f"Изображение {sat_name} не найдено")

        text_path = os.path.join(SATELLITE_DIR, f"{sat_name}.txt")
        self._load_text(text_path)
        self.statusBar().showMessage(f"Загружено: {sat_name}", 3000)

    def _load_text(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            processed = text.replace("\n", "<br>")
            self.sat_info.setText(
                f"<html><body style='color:#2c3e50; line-height:1.6;'>{processed}</body></html>"
            )
        except FileNotFoundError:
            self.sat_info.setText("Описание отсутствует")

    # ================================================================
    #  Кнопка 2: Расчёт одиночного РЛИ
    # ================================================================

    def _calc_single_rli(self):
        self._on_config_changed()
        self.info_label.setText("Расчёт одиночного РЛИ...")
        self.statusBar().showMessage("Расчёт одиночного РЛИ...")

        filename = get_matrix_filename(self.cfg.satellite)

        proc = DataProcessor(
            filename=filename,
            method=self.cfg.method,
            f_c=self.cfg.center_frequency,
            Xmax=self.cfg.survey_length,
            Ymax=self.cfg.survey_width,
            spectr_w=self.cfg.spectrum_width,
            ang=self.cfg.beam_width,
            range_m=self.cfg.range_m,
            nifft_size=self.cfg.nifft_size,
        )
        # вычисляем первый кадр синхронно
        frame = proc.compute_single()
        if frame:
            self._display_frame(frame)
            self.info_label.setText("Одиночный РЛИ готов")
            self.statusBar().showMessage("Одиночный РЛИ готов", 3000)

    # ================================================================
    #  Кнопка 3: Расчёт потока РЛИ
    # ================================================================

    def _calc_stream(self):
        self._on_config_changed()
        self.info_label.setText("Расчёт потока РЛИ...")
        self.statusBar().showMessage("Расчёт потока РЛИ...")

        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()

        filename = get_matrix_filename(self.cfg.satellite)
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radioimage")

        self.processor = DataProcessor(
            filename=filename,
            method=self.cfg.method,
            f_c=self.cfg.center_frequency,
            Xmax=self.cfg.survey_length,
            Ymax=self.cfg.survey_width,
            spectr_w=self.cfg.spectrum_width,
            ang=self.cfg.beam_width,
            range_m=self.cfg.range_m,
            nifft_size=self.cfg.nifft_size,
            save_dir=save_dir,
        )

        self.progress_bar.setRange(0, NUM_IMAGES)
        self.progress_bar.setValue(0)

        self.processor.frame_progress.connect(self._on_frame_progress)
        self.processor.start()

    def _on_frame_progress(self, i):
        self.progress_bar.setValue(i)
        self.info_label.setText(f"Расчёт: {i}/{NUM_IMAGES}")

    # ================================================================
    #  Кнопка 4: Вывести результаты (поток)
    # ================================================================

    def _show_results(self):
        """Показать сохранённые изображения потоком."""
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radioimage")
        if not os.path.exists(save_dir):
            self.info_label.setText("Нет сохранённых результатов")
            return

        self._frame_index = 0
        self._save_dir = save_dir
        self.info_label.setText("Вывод кадров...")
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._show_next_result_from_file)
        self._render_timer.start(75)
        self.statusBar().showMessage("Вывод результатов...")

    def _show_next_result_from_file(self):
        if self._frame_index >= NUM_IMAGES:
            self._render_timer.stop()
            self.info_label.setText("Поток завершён")
            self.statusBar().showMessage("Поток завершён", 3000)
            return

        frame_dir = os.path.join(self._save_dir, f"frame_{self._frame_index + 1:03d}")
        field_path = os.path.join(frame_dir, "field.png")
        rli_path = os.path.join(frame_dir, "rli.png")

        if os.path.exists(field_path) and os.path.exists(rli_path):
            field_array = cv2.imread(field_path)
            rli_array = cv2.imread(rli_path)
            
            field_array = cv2.cvtColor(field_array, cv2.COLOR_BGR2RGB)
            rli_array = cv2.cvtColor(rli_array, cv2.COLOR_BGR2RGB)
            
            self.pg_field_img.setImage(field_array)
            self.pg_result_img.setImage(rli_array)

        self.progress_bar.setValue(self._frame_index + 1)
        self._frame_index += 1

    def _set_field(self, data, y_label, x_label, x_range, y_range):
        """Установить данные и оси поля обратного рассеяния."""
        h, w = data.shape
        x_min, x_max = x_range
        y_min, y_max = y_range
        self.pg_field_img.setImage(data)
        self.pg_field_img.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        self.pg_field.setLabel("left", y_label)
        self.pg_field.setLabel("bottom", x_label)
        self.pg_field_view.autoRange()

    def _set_result(self, data, x_range, y_range):
        """Установить данные и оси РЛИ."""
        h, w = data.shape
        x_min, x_max = x_range
        y_min, y_max = y_range
        self.pg_result_img.setImage(data)
        self.pg_result_img.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        self.pg_result.setLabel("left", "Y, м")
        self.pg_result.setLabel("bottom", "X, м")
        self.pg_result_view.autoRange()

    # ================================================================
    #  Отображение одного кадра
    # ================================================================

    def _display_frame(self, mat):
        method = self.cfg.method
        if method == "стандартный":
            h = SPEED_OF_LIGHT / (2.0 * self.cfg.center_frequency * 1e9)
            field_data = abs(np.rot90(mat[0], 2))
            img_data = abs(np.rot90(mat[3], 2))
            self._set_field(
                field_data,
                y_label="Частота, Гц", x_label="Ширина ДН, радиан",
                x_range=(mat[2][0], mat[2][-1]), y_range=(mat[1][0], mat[1][-1]),
            )
            self._set_result(
                img_data,
                x_range=(mat[4][0], mat[4][-1]),
                y_range=(mat[5][0] * h, mat[5][-1] * h),
            )
        elif method == "с полярным переформатированием":
            field_data = abs(mat[0])
            img_data = abs(mat[3] / (mat[4] * mat[5]))
            self._set_field(
                field_data,
                y_label="Kx, рад/м", x_label="Ky, рад/м",
                x_range=(np.min(mat[2]), np.max(mat[2])),
                y_range=(np.min(mat[1]), np.max(mat[1])),
            )
            self._set_result(
                img_data,
                x_range=(np.min(mat[7]), np.max(mat[7])),
                y_range=(np.min(mat[6]), np.max(mat[6])),
            )

    # ================================================================
    #  Ресайз
    # ================================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "image_label_sat") and self.image_label_sat.pixmap():
            original = self.image_label_sat.pixmap()
            self.image_label_sat.setPixmap(
                original.scaled(self.image_label_sat.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
