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
    QTabWidget, QCheckBox, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QThread
from PyQt5.QtGui import QPixmap

from config import AppConfig
from filenames import get_matrix_filename
from processing import DataProcessor
from ui.styles import APP_STYLE

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SATELLITE_DIR = os.path.join(BASE_DIR, "sat_info_p")
SPEED_OF_LIGHT = 3e8


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cfg = AppConfig.load()
        self.processor = None
        self._frame_index = 0
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("IRSA — ISAR Radar Image Simulator")
        self.resize(1400, 800)
        self.setMinimumSize(1000, 600)
        self.setStyleSheet(APP_STYLE)
        pg.setConfigOptions(imageAxisOrder="row-major")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        # Вертикальный сплиттер: параметры (верх) | результаты (низ)
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(3)

        # Верхняя часть: спутник + параметры + кнопки
        main_splitter.addWidget(self._build_upper_section())
        # Нижняя часть: вкладки с результатами
        main_splitter.addWidget(self._build_tab_widget())
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([280, 520])

        root_layout.addWidget(main_splitter)
        self.statusBar().showMessage("Готово к работе")

    # ================================================================
    #  Верхняя секция: спутник | параметры | кнопки
    # ================================================================

    def _build_upper_section(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._build_satellite_panel())
        splitter.addWidget(self._build_params_panel())
        splitter.addWidget(self._build_buttons_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([300, 500, 180])
        return splitter

    # ---------- Спутник ----------

    def _build_satellite_panel(self):
        grp = QGroupBox("Исследуемый спутник")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self.image_label_sat = QLabel("Нажмите «Показать КА»")
        self.image_label_sat.setAlignment(Qt.AlignCenter)
        self.image_label_sat.setMinimumHeight(80)
        self.image_label_sat.setMaximumHeight(120)
        self.image_label_sat.setStyleSheet(
            "background-color: #fafafa; border: 1px dashed #bdc3c7;"
            "border-radius: 6px; padding: 4px;"
        )
        layout.addWidget(self.image_label_sat)

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

    # ---------- Параметры ----------

    def _build_params_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Цель и метод
        layout.addWidget(self._section_label("Цель и метод"))
        row1 = QVBoxLayout()
        row1.setSpacing(4)
        sat_lbl, self.cb_satellite = self._make_combo_widget(
            "Спутник",
            ["cloudSAT", "calipso", "ICEsat2", "LRO", "solarB"],
            self.cfg.satellite,
        )
        method_lbl, self.cb_method = self._make_combo_widget(
            "Метод обработки",
            ["стандартный", "с полярным переформатированием"],
            self.cfg.method,
        )
        combo_row = QHBoxLayout()
        combo_row.addWidget(sat_lbl)
        combo_row.addWidget(self.cb_satellite)
        combo_row.addWidget(method_lbl)
        combo_row.addWidget(self.cb_method)
        row1.addLayout(combo_row)
        layout.addLayout(row1)

        # Параметры радара
        layout.addWidget(self._section_label("Параметры радара"))
        radar_grid = QGridLayout()
        radar_grid.setSpacing(4)
        self.le_frequency = self._make_input_widget(
            "Частота, ГГц", self.cfg.center_frequency,
            tooltip="Несущая частота радара (ГГц). Типично: 8-10 ГГц для ISAR.",
        )
        self.le_spectrum = self._make_input_widget(
            "Спектр, ГГц", self.cfg.spectrum_width,
            tooltip="Полоса частот SFCW (ГГц). Определяет разрешение по дальности: Δr = c/(2·B).",
        )
        self.le_nifft = self._make_input_widget(
            "Размер FFT", self.cfg.nifft_size,
            tooltip="Размер IFFT (степень 2). Больше = лучше разрешение сетки в k-space.",
        )
        self.le_beam = self._make_input_widget(
            "Ширина ДН, град", self.cfg.beam_width,
            tooltip="Ширина диаграммы направленности (°). Используется для авто-расчёта ω при omega=0.",
        )
        self.le_range = self._make_input_widget(
            "Дальность, км", self.cfg.range_km,
            tooltip="Дальность до объекта (км). Влияет на опорный сигнал дешифровки (dechirp).",
        )
        radar_grid.addWidget(self.le_frequency[0], 0, 0); radar_grid.addWidget(self.le_frequency[1], 0, 1)
        radar_grid.addWidget(self.le_spectrum[0], 0, 2); radar_grid.addWidget(self.le_spectrum[1], 0, 3)
        radar_grid.addWidget(self.le_nifft[0], 1, 0); radar_grid.addWidget(self.le_nifft[1], 1, 1)
        radar_grid.addWidget(self.le_beam[0], 1, 2); radar_grid.addWidget(self.le_beam[1], 1, 3)
        radar_grid.addWidget(self.le_range[0], 2, 0); radar_grid.addWidget(self.le_range[1], 2, 1)
        layout.addLayout(radar_grid)

        # Параметры обзора
        layout.addWidget(self._section_label("Обзор"))
        survey_grid = QGridLayout()
        survey_grid.setSpacing(4)
        self.le_survey_length = self._make_input_widget(
            "Длина, м", self.cfg.survey_length,
            tooltip="Размер сцены по дальности (м). Используется для проектирования декартовой сетки.",
        )
        self.le_survey_width = self._make_input_widget(
            "Ширина, м", self.cfg.survey_width,
            tooltip="Размер сцены по азимуту (м). Используется для антиалиасинга по азимуту.",
        )
        survey_grid.addWidget(self.le_survey_length[0], 0, 0); survey_grid.addWidget(self.le_survey_length[1], 0, 1)
        survey_grid.addWidget(self.le_survey_width[0], 0, 2); survey_grid.addWidget(self.le_survey_width[1], 0, 3)
        layout.addLayout(survey_grid)

        # Кинематика цели
        layout.addWidget(self._section_label("Кинематика цели"))
        kin_grid = QGridLayout()
        kin_grid.setSpacing(4)
        self.le_V = self._make_input_widget(
            "V, м/с", self.cfg.V,
            tooltip="Поступательная скорость цели (м/с). 0 = Turntable (только вращение).",
        )
        self.le_alpha = self._make_input_widget(
            "α, град", self.cfg.alpha,
            tooltip="Угол направления движения (°). Используется при V != 0.",
        )
        self.le_omega = self._make_input_widget(
            "ω, рад/с", self.cfg.omega,
            tooltip="Угловая скорость вращения (рад/с). 0 = авто из beam_width.\n"
                    "Δθ = ω · N · PRI. При Δθ ≥ 3° рекомендуется полярное переформатирование.",
        )
        self.le_pri = self._make_input_widget(
            "PRI, с", self.cfg.pri,
            tooltip="Период повторения импульсов (с). T_obs = PRI · N.",
        )
        self.le_num_pulses = self._make_input_widget(
            "Импульсов", self.cfg.num_pulses,
            tooltip="Число импульсов (= размер азимутальной оси РЛИ).\n"
                    "Рекомендуется 1000 для качественного фокусированного изображения.",
        )
        kin_grid.addWidget(self.le_V[0], 0, 0); kin_grid.addWidget(self.le_V[1], 0, 1)
        kin_grid.addWidget(self.le_alpha[0], 0, 2); kin_grid.addWidget(self.le_alpha[1], 0, 3)
        kin_grid.addWidget(self.le_omega[0], 1, 0); kin_grid.addWidget(self.le_omega[1], 1, 1)
        kin_grid.addWidget(self.le_pri[0], 1, 2); kin_grid.addWidget(self.le_pri[1], 1, 3)
        kin_grid.addWidget(self.le_num_pulses[0], 2, 0); kin_grid.addWidget(self.le_num_pulses[1], 2, 1)
        layout.addLayout(kin_grid)

        # Параметры обработки
        layout.addWidget(self._section_label("Обработка"))
        proc_layout = QHBoxLayout()
        proc_layout.setSpacing(12)

        self.chk_mocomp = QCheckBox("MOCOMP")
        self.chk_mocomp.setChecked(self.cfg.use_mocomp)
        self.chk_mocomp.stateChanged.connect(self._on_config_changed)
        proc_layout.addWidget(self.chk_mocomp)

        win_lbl, self.cb_window = self._make_combo_widget(
            "Окно", ["none", "hamming", "hann"], self.cfg.window,
        )
        disp_lbl, self.cb_display = self._make_combo_widget(
            "Отображение", ["linear", "dB"], self.cfg.display_mode,
        )
        proc_layout.addWidget(win_lbl)
        proc_layout.addWidget(self.cb_window)
        proc_layout.addWidget(disp_lbl)
        proc_layout.addWidget(self.cb_display)
        proc_layout.addStretch()
        layout.addLayout(proc_layout)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ---------- Кнопки ----------

    def _build_buttons_panel(self):
        grp = QGroupBox("Управление")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self.btn_show_sat = self._make_button("Показать КА", self._show_satellite)
        layout.addWidget(self.btn_show_sat)

        self.btn_single = self._make_button("Рассчитать РЛИ", self._calc_single_rli, style_class="btn_action")
        layout.addWidget(self.btn_single)

        self.btn_stream = self._make_button("Запустить поток", self._calc_stream, style_class="btn_secondary")
        layout.addWidget(self.btn_stream)

        self.btn_stop = self._make_button("Остановить", self._stop_stream)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        layout.addSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.info_label = QLabel("")
        self.info_label.setObjectName("status_info")
        layout.addWidget(self.info_label)

        layout.addStretch()
        return grp

    # ================================================================
    #  Нижняя секция: QTabWidget с результатами
    # ================================================================

    def _build_tab_widget(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Вкладка 0: Сигнал
        self.tabs.addTab(self._build_signal_tab(), "Сигнал")
        # Вкладка 1: Профили дальности
        self.tabs.addTab(self._build_range_tab(), "Профили дальности")
        # Вкладка 2: Компенсация движения
        self.tabs.addTab(self._build_mocomp_tab(), "Компенсация движения")
        # Вкладка 3: РЛИ
        self.tabs.addTab(self._build_isar_tab(), "РЛИ")
        # Вкладка 4: Поток кадров
        self.tabs.addTab(self._build_stream_tab(), "Поток кадров")

        return self.tabs

    def _build_signal_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Левая часть: параметры
        left = QVBoxLayout()
        left.addWidget(self._section_label("Параметры сигнала SFCW"))
        self.signal_table = QTableWidget()
        self.signal_table.setColumnCount(2)
        self.signal_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.signal_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.signal_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.signal_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.signal_table.setSelectionMode(QTableWidget.NoSelection)
        self.signal_table.verticalHeader().setVisible(False)
        self.signal_table.setMinimumWidth(280)
        left.addWidget(self.signal_table)
        left.addStretch()
        layout.addLayout(left, stretch=1)

        # Правая часть: два графика
        right = QVBoxLayout()

        right.addWidget(self._section_label("Временная область — SFCW-импульс"))
        self.pg_signal_time = pg.PlotWidget(background="#fafafa")
        self.pg_signal_time.setLabel("left", "s(t)")
        self.pg_signal_time.setLabel("bottom", "Время, мкс")
        self.pg_signal_time.setTitle("Переданный SFCW-сигнал: частотная лесенка")
        self.signal_time_curve = self.pg_signal_time.plot(
            pen=pg.mkPen(color="#2980b9", width=1.2)
        )
        right.addWidget(self.pg_signal_time, stretch=1)

        right.addWidget(self._section_label("Частотная область — спектр сигнала"))
        self.pg_signal_spec = pg.PlotWidget(background="#fafafa")
        self.pg_signal_spec.setLabel("left", "|S(f)|")
        self.pg_signal_spec.setLabel("bottom", "Частота, ГГц")
        self.pg_signal_spec.setTitle("Спектр SFCW-импульса")
        self.signal_spec_curve = self.pg_signal_spec.plot(
            pen=pg.mkPen(color="#c0392b", width=1.5)
        )
        right.addWidget(self.pg_signal_spec, stretch=1)

        layout.addLayout(right, stretch=2)
        return widget

    def _update_signal_tab(self, E, f_r, num_pulses):
        """Обновить вкладку «Сигнал» данными из DataProcessor."""
        if E is None or f_r is None:
            return

        Nf = E.shape[0]
        Np = E.shape[1]
        c = SPEED_OF_LIGHT
        f_c = f_r[Nf // 2] if Nf > 0 else 0
        bw = f_r[-1] - f_r[0] if Nf > 1 else 0
        df = f_r[1] - f_r[0] if Nf > 1 else 0
        wavelength = c / f_c if f_c > 0 else 0
        dr = c / (2 * bw) if bw > 0 else 0
        pri = self.cfg.pri
        T_obs = Np * pri
        omega = self.cfg.omega
        dx = wavelength / (2 * omega * T_obs) if omega > 0 and T_obs > 0 else 0

        params = [
            ("Несущая частота f_c", f"{f_c / 1e9:.3f} ГГц"),
            ("Длина волны λ", f"{wavelength * 100:.2f} см"),
            ("Полоса частот B", f"{bw / 1e9:.3f} ГГц"),
            ("Число частотных шагов Nf", f"{Nf}"),
            ("Шаг частоты Δf", f"{df / 1e6:.2f} МГц"),
            ("Разрешение по дальности Δr", f"{dr:.3f} м"),
            ("", ""),
            ("Число импульсов N", f"{Np}"),
            ("PRI", f"{pri} с"),
            ("Время наблюдения T_obs", f"{T_obs:.2f} с"),
            ("Угловая скорость ω", f"{omega:.6f} рад/с"),
            ("Разрешение по азимуту Δx", f"{dx:.3f} м"),
            ("", ""),
            ("Размер матрицы E", f"{Nf} × {Np}"),
        ]

        self.signal_table.setRowCount(len(params))
        for row, (name, value) in enumerate(params):
            self.signal_table.setItem(row, 0, QTableWidgetItem(name))
            self.signal_table.setItem(row, 1, QTableWidgetItem(value))

        # --- Временная область: SFCW-лесенка ---
        tau_step = 1.0 / bw if bw > 0 else 1e-6
        t_total = Nf * tau_step
        n_pts_per_step = 300
        t_all = np.linspace(0, t_total, Nf * n_pts_per_step, endpoint=False)
        s_time = np.zeros_like(t_all)
        for m in range(Nf):
            f_m = f_r[m]
            t_start = m * tau_step
            t_end = (m + 1) * tau_step
            mask = (t_all >= t_start) & (t_all < t_end)
            s_time[mask] = np.sin(2 * np.pi * f_m * t_all[mask])
        self.signal_time_curve.setData(t_all * 1e6, s_time)
        self.pg_signal_time.setTitle(f"SFCW-импульс ({t_total * 1e6:.1f} мкс, Nf={Nf} шагов)")

        # --- Частотная область: спектр SFCW-сигнала ---
        n_fft = len(s_time)
        S_f = np.abs(np.fft.fft(s_time))
        S_f = np.fft.fftshift(S_f)
        freq_axis = np.fft.fftfreq(n_fft, d=(t_all[1] - t_all[0]))
        freq_axis = np.fft.fftshift(freq_axis)
        self.signal_spec_curve.setData(freq_axis / 1e9, S_f)

    def _build_range_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.pg_range = pg.PlotWidget(background="#fafafa")
        self.pg_range.setLabel("left", "Дальность, м")
        self.pg_range.setLabel("bottom", "Импульс №")
        self.pg_range_img = pg.ImageItem()
        self.pg_range.getPlotItem().getViewBox().addItem(self.pg_range_img)

        layout.addWidget(self.pg_range)
        return widget

    def _build_mocomp_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()

        # График сдвигов огибающей
        self.pg_shifts = pg.PlotWidget(background="#fafafa")
        self.pg_shifts.setLabel("left", "Сдвиг, бин")
        self.pg_shifts.setLabel("bottom", "Импульс №")
        self.pg_shifts.setTitle("Выравнивание огибающей")
        self.shifts_curve = self.pg_shifts.plot(pen=pg.mkPen(color="#3498db", width=2))
        top_row.addWidget(self.pg_shifts)

        # График фазовых ошибок
        self.pg_phase = pg.PlotWidget(background="#fafafa")
        self.pg_phase.setLabel("left", "Фаза, рад")
        self.pg_phase.setLabel("bottom", "Импульс №")
        self.pg_phase.setTitle("Фазовые ошибки")
        self.phase_curve = self.pg_phase.plot(pen=pg.mkPen(color="#e74c3c", width=2))
        top_row.addWidget(self.pg_phase)

        layout.addLayout(top_row, stretch=1)

        # Heatmaps до/после
        bottom_row = QHBoxLayout()

        self.pg_mocomp_before = pg.PlotWidget(background="#fafafa")
        self.pg_mocomp_before.setLabel("left", "Дальность, м")
        self.pg_mocomp_before.setLabel("bottom", "Импульс №")
        self.pg_mocomp_before.setTitle("До MOCOMP")
        self.mocomp_before_img = pg.ImageItem()
        self.pg_mocomp_before.getPlotItem().getViewBox().addItem(self.mocomp_before_img)
        bottom_row.addWidget(self.pg_mocomp_before)

        self.pg_mocomp_after = pg.PlotWidget(background="#fafafa")
        self.pg_mocomp_after.setLabel("left", "Дальность, м")
        self.pg_mocomp_after.setLabel("bottom", "Импульс №")
        self.pg_mocomp_after.setTitle("После MOCOMP")
        self.mocomp_after_img = pg.ImageItem()
        self.pg_mocomp_after.getPlotItem().getViewBox().addItem(self.mocomp_after_img)
        bottom_row.addWidget(self.pg_mocomp_after)

        layout.addLayout(bottom_row, stretch=1)

        return widget

    def _build_isar_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.pg_isar = pg.PlotWidget(background="#fafafa")
        self.pg_isar.setLabel("left", "Y, м")
        self.pg_isar.setLabel("bottom", "X, м")
        self.pg_isar.setTitle("ISAR РЛИ")
        self.isar_img = pg.ImageItem()
        self.pg_isar.getPlotItem().getViewBox().addItem(self.isar_img)

        layout.addWidget(self.pg_isar)
        return widget

    def _build_stream_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.pg_stream = pg.PlotWidget(background="#fafafa")
        self.pg_stream.setLabel("left", "Y, м")
        self.pg_stream.setLabel("bottom", "X, м")
        self.pg_stream_img = pg.ImageItem()
        self.pg_stream.getPlotItem().getViewBox().addItem(self.pg_stream_img)

        layout.addWidget(self.pg_stream)
        return widget

    # ================================================================
    #  Фабрики виджетов
    # ================================================================

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _make_combo_widget(self, label_text, items, current):
        lbl = QLabel(label_text)
        lbl.setObjectName("status_info")
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(self._on_config_changed)
        return lbl, combo

    def _make_input_widget(self, label_text, value, tooltip=None):
        lbl = QLabel(label_text)
        lbl.setObjectName("status_info")
        le = QLineEdit(str(value))
        le.setMaximumWidth(80)
        le.textChanged.connect(self._on_config_changed)
        if tooltip:
            le.setToolTip(tooltip)
            lbl.setToolTip(tooltip)
        return lbl, le

    def _make_button(self, text, slot, style_class=None):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        if style_class:
            btn.setObjectName(style_class)
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
        self.cfg.center_frequency = self._safe_float(self.le_frequency[1], self.cfg.center_frequency)
        self.cfg.spectrum_width = self._safe_float(self.le_spectrum[1], self.cfg.spectrum_width)
        self.cfg.nifft_size = self._safe_int(self.le_nifft[1], self.cfg.nifft_size)
        self.cfg.beam_width = self._safe_float(self.le_beam[1], self.cfg.beam_width)
        self.cfg.survey_length = self._safe_float(self.le_survey_length[1], self.cfg.survey_length)
        self.cfg.survey_width = self._safe_float(self.le_survey_width[1], self.cfg.survey_width)
        self.cfg.range_km = self._safe_float(self.le_range[1], self.cfg.range_km)

        # Кинематика
        self.cfg.V = self._safe_float(self.le_V[1], self.cfg.V)
        self.cfg.alpha = self._safe_float(self.le_alpha[1], self.cfg.alpha)
        self.cfg.omega = self._safe_float(self.le_omega[1], self.cfg.omega)
        self.cfg.pri = self._safe_float(self.le_pri[1], self.cfg.pri)
        self.cfg.num_pulses = self._safe_int(self.le_num_pulses[1], self.cfg.num_pulses)

        # Обработка
        self.cfg.use_mocomp = self.chk_mocomp.isChecked()
        self.cfg.window = self.cb_window.currentText()
        self.cfg.display_mode = self.cb_display.currentText()

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
    #  Кнопка 2: Рассчитать РЛИ (новый конвейер)
    # ================================================================

    def _calc_single_rli(self):
        self._on_config_changed()
        self.info_label.setText("Расчёт РЛИ...")
        self.statusBar().showMessage("Расчёт РЛИ (конвейер: Raw→Range→MOCOMP→Azimuth)...")

        try:
            filename = get_matrix_filename(self.cfg.satellite)
        except Exception as e:
            self._on_error(f"Ошибка конфигурации: {e}")
            return

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
            V=self.cfg.V,
            alpha=self.cfg.alpha,
            omega=self.cfg.omega,
            pri=self.cfg.pri,
            num_pulses=self.cfg.num_pulses,
            use_mocomp=self.cfg.use_mocomp,
            window=self.cfg.window,
            display_mode=self.cfg.display_mode,
        )

        # Подключаем сигналы промежуточных результатов
        self.processor.range_profiles_ready.connect(self._on_range_profiles)
        self.processor.mocomp_envelope_ready.connect(self._on_mocomp_envelope)
        self.processor.mocomp_phase_ready.connect(self._on_mocomp_phase)
        self.processor.mocomp_before_after.connect(self._on_mocomp_before_after)
        self.processor.isar_ready.connect(self._on_isar_ready)
        self.processor.signal_data_ready.connect(self._on_signal_data)
        self.processor.progress_message.connect(self._on_progress_message)
        self.processor.error_occurred.connect(self._on_error)

        # Запускаем в потоке
        self.btn_single.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.processor.finished.connect(self._on_calc_finished)
        self.processor.start()

    def _on_calc_finished(self):
        self.btn_single.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.info_label.setObjectName("status_success")
        self.info_label.style().unpolish(self.info_label)
        self.info_label.style().polish(self.info_label)
        self.info_label.setText("РЛИ готов")
        self.statusBar().showMessage("Расчёт завершён", 3000)
        # Переключаемся на вкладку РЛИ
        self.tabs.setCurrentIndex(3)

    # ================================================================
    #  Слоты для промежуточных результатов
    # ================================================================

    def _on_signal_data(self, E, f_r):
        self._update_signal_tab(E, f_r, self.cfg.num_pulses)

    def _on_range_profiles(self, P_abs, range_axis, dr):
        self.pg_range_img.setImage(P_abs)
        self.pg_range_img.setRect(QRectF(
            0, range_axis[0],
            P_abs.shape[1], range_axis[-1] - range_axis[0],
        ))
        vb = self.pg_range.getPlotItem().getViewBox()
        vb.enableAutoRange()
        vb.updateAutoRange()
        self.tabs.setTabText(1, f"Профили дальности (Δr={dr:.3f} м)")
        self.tabs.setCurrentIndex(1)

    def _on_mocomp_envelope(self, shifts, num_pulses):
        x = np.arange(num_pulses)
        self.shifts_curve.setData(x, shifts)
        self.pg_shifts.getPlotItem().getViewBox().enableAutoRange()

    def _on_mocomp_phase(self, phase_errors):
        x = np.arange(len(phase_errors))
        self.phase_curve.setData(x, phase_errors)
        self.pg_phase.getPlotItem().getViewBox().enableAutoRange()

    def _on_mocomp_before_after(self, P_abs_before, P_abs_after, range_axis):
        self.mocomp_before_img.setImage(P_abs_before)
        self.mocomp_before_img.setRect(QRectF(
            0, range_axis[0],
            P_abs_before.shape[1], range_axis[-1] - range_axis[0],
        ))
        self.mocomp_after_img.setImage(P_abs_after)
        self.mocomp_after_img.setRect(QRectF(
            0, range_axis[0],
            P_abs_after.shape[1], range_axis[-1] - range_axis[0],
        ))
        vb_before = self.pg_mocomp_before.getPlotItem().getViewBox()
        vb_before.enableAutoRange()
        vb_before.updateAutoRange()
        vb_after = self.pg_mocomp_after.getPlotItem().getViewBox()
        vb_after.enableAutoRange()
        vb_after.updateAutoRange()

    def _on_isar_ready(self, amplitude, range_axis, azimuth_axis):
        if self.cfg.display_mode == "dB":
            self.isar_img.setImage(amplitude, levels=(-40, 0))
        else:
            self.isar_img.setImage(amplitude, autoLevels=True)
        self.isar_img.setRect(QRectF(
            azimuth_axis[0], range_axis[0],
            azimuth_axis[-1] - azimuth_axis[0], range_axis[-1] - range_axis[0],
        ))
        vb = self.pg_isar.getPlotItem().getViewBox()
        vb.enableAutoRange()
        vb.updateAutoRange()
        method_label = "polar reformat" if self.cfg.method == "с полярным переформатированием" else "standard"
        self.pg_isar.setTitle(f"ISAR ({method_label})")

    def _on_progress_message(self, msg):
        logger.info(msg)
        self.info_label.setObjectName("status_info")
        self.info_label.style().unpolish(self.info_label)
        self.info_label.style().polish(self.info_label)
        self.info_label.setText(msg)

    def _on_error(self, msg):
        logger.error(msg)
        self.info_label.setObjectName("status_error")
        self.info_label.style().unpolish(self.info_label)
        self.info_label.style().polish(self.info_label)
        self.info_label.setText(f"Ошибка: {msg}")
        self.statusBar().showMessage("Ошибка расчёта", 5000)
        self.btn_single.setEnabled(True)
        self.btn_stream.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    # ================================================================
    #  Кнопка 3: Запустить поток (старый конвейер, покадровый)
    # ================================================================

    def _calc_stream(self):
        self._on_config_changed()
        self.info_label.setText("Расчёт потока РЛИ...")
        self.statusBar().showMessage("Расчёт потока РЛИ...")

        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()

        try:
            filename = get_matrix_filename(self.cfg.satellite)
        except Exception as e:
            self._on_error(f"Ошибка конфигурации: {e}")
            return

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
            V=self.cfg.V,
            alpha=self.cfg.alpha,
            omega=self.cfg.omega,
            pri=self.cfg.pri,
            num_pulses=self.cfg.num_pulses,
            use_mocomp=self.cfg.use_mocomp,
            window=self.cfg.window,
            display_mode=self.cfg.display_mode,
            save_dir=save_dir,
            mode="stream",
        )

        self.progress_bar.setRange(0, self.cfg.num_pulses)
        self.progress_bar.setValue(0)
        self.btn_stream.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.processor.frame_progress.connect(self._on_frame_progress)
        self.processor.progress_message.connect(self._on_progress_message)
        self.processor.error_occurred.connect(self._on_error)
        self.processor.finished.connect(self._on_stream_finished)
        self.processor.start()

    def _on_frame_progress(self, i):
        self.progress_bar.setValue(i)
        self.info_label.setText(f"Расчёт: {i}/{self.cfg.num_pulses}")

    def _on_stream_finished(self):
        self.btn_stream.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.info_label.setText("Расчёт завершён. Вывод результатов...")
        self.statusBar().showMessage("Поток завершён", 3000)
        self._show_stream_results()

    def _stop_stream(self):
        if self.processor:
            self.processor.stop()
        self.btn_stream.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.info_label.setText("Остановлено")
        self.statusBar().showMessage("Расчёт остановлен", 3000)

    # ================================================================
    #  Кнопка 4: Вывести результаты потока (таб "Поток кадров")
    # ================================================================

    def _show_stream_results(self):
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radioimage")
        if not os.path.exists(save_dir):
            self.info_label.setText("Нет сохранённых результатов")
            return

        self._frame_index = 0
        self._save_dir = save_dir
        self.info_label.setText("Вывод кадров...")
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._show_next_stream_frame)
        self._render_timer.start(75)
        self.tabs.setCurrentIndex(4)
        self.statusBar().showMessage("Вывод результатов...")

    def _show_next_stream_frame(self):
        if self._frame_index >= self.cfg.num_pulses:
            self._render_timer.stop()
            self.info_label.setText("Поток завершён")
            self.statusBar().showMessage("Поток завершён", 3000)
            return

        frame_dir = os.path.join(self._save_dir, f"frame_{self._frame_index + 1:03d}")
        rli_path = os.path.join(frame_dir, "rli.png")

        if os.path.exists(rli_path):
            rli_array = cv2.imread(rli_path)
            rli_array = cv2.cvtColor(rli_array, cv2.COLOR_BGR2RGB)
            self.pg_stream_img.setImage(rli_array)
            vb = self.pg_stream.getPlotItem().getViewBox()
            vb.enableAutoRange()
            vb.updateAutoRange()

        self.progress_bar.setValue(self._frame_index + 1)
        self.info_label.setText(f"Кадр {self._frame_index + 1}/{self.cfg.num_pulses}")
        self._frame_index += 1

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
