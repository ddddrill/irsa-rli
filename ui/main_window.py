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
    QSlider,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QThread
from PyQt5.QtGui import QPixmap

from config import AppConfig
from paths import resource_path, app_dir
from filenames import get_matrix_filename
from processing import DataProcessor
from processing.windows import WINDOW_NAMES
from processing.tf_analysis import rid_frame, rid_marginal, wigner_ville
from ui.styles import APP_STYLE
from ui.help_dialog import HelpDialog

logger = logging.getLogger(__name__)

SATELLITE_DIR = resource_path("sat_info_p")
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
        self.le_snr = self._make_input_widget(
            "SNR, дБ", self.cfg.snr_db,
            tooltip="Отношение сигнал/шум (дБ). inf = без шума.\n"
                    "20 = сильный сигнал, 0 = сигнал=шум, -5 = шум сильнее сигнала.",
        )
        radar_grid.addWidget(self.le_snr[0], 2, 2); radar_grid.addWidget(self.le_snr[1], 2, 3)
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

        # Модель цели + 3D-кинематика (Tier 2)
        layout.addWidget(self._section_label("Модель цели"))
        model_grid = QGridLayout()
        model_grid.setSpacing(4)

        model_lbl, self.cb_target_model = self._make_combo_widget(
            "Модель", ["точечная", "формы (json)"],
            getattr(self.cfg, "target_model", "точечная"),
        )
        self.cb_target_model.setToolTip(
            "Метка пресета: «формы (json)» уточняет порядок загрузки "
            "рассеивателей; геометрия остаётся точечно-рассеивательной "
            "(3D-модель форм не реализована в 2D-конвейере).")
        self.cb_target_model.currentTextChanged.connect(self._on_config_changed)
        model_grid.addWidget(model_lbl, 0, 0); model_grid.addWidget(self.cb_target_model, 0, 1)

        self.le_omega_axis_y = self._make_input_widget(
            "Ось ω (y)", self.cfg.omega_axis_y,
            tooltip="Y-компонента оси вращения в СК цели (X=0 всегда).\n"
                    "(0, 0, 1) = ось Z (по умолчанию); (0, 1, 0) = ось Y.\n"
                    "Не реализовано в 2D-конвейере (плоскость LOS×кросс-рейндж);"
                    " требует 3D-кинематики (ω_eff = ω·cos β).",
        )
        self.le_omega_axis_z = self._make_input_widget(
            "Ось ω (z)", self.cfg.omega_axis_z,
            tooltip="Z-компонента оси вращения в СК цели.\n"
                    "Не реализовано в 2D-конвейере (плоскость LOS×кросс-рейндж);"
                    " требует 3D-кинематики (ω_eff = ω·cos β).",
        )
        model_grid.addWidget(self.le_omega_axis_y[0], 0, 2); model_grid.addWidget(self.le_omega_axis_y[1], 0, 3)
        model_grid.addWidget(self.le_omega_axis_z[0], 1, 2); model_grid.addWidget(self.le_omega_axis_z[1], 1, 3)

        self.le_precession_rate = self._make_input_widget(
            "Прецессия, рад/с", self.cfg.precession_rate,
            tooltip="Частота прецессии оси вращения (рад/с). 0 = нет прецессии.\n"
                    "Не реализовано в 2D-конвейере — требует 3D-кинематики;"
                    " молчаливый no-op отключён (см. «Не реализовано (2D)» в"
                    " PARAMETERS_GUIDE).",
        )
        self.le_precession_angle = self._make_input_widget(
            "Прецессия, град", self.cfg.precession_angle_deg,
            tooltip="Полуразворот конуса прецессии (град).\n"
                    "Не реализовано в 2D-конвейере — требует 3D-кинематики;"
                    " молчаливый no-op отключён (см. «Не реализовано (2D)» в"
                    " PARAMETERS_GUIDE).",
        )
        model_grid.addWidget(self.le_precession_rate[0], 1, 0); model_grid.addWidget(self.le_precession_rate[1], 1, 1)
        model_grid.addWidget(self.le_precession_angle[0], 2, 0); model_grid.addWidget(self.le_precession_angle[1], 2, 1)
        # F8: поля 3D-кинематики (прецессия, ось ω y/z) молча игнорировались
        # 2D-конвейером (_make_target их не передаёт в Target) — молчаливый
        # no-op хуже отсутствия фичи. Поля недоступны; параметры остаются в
        # API DataProcessor (обратная совместимость), физика не менялась.
        _NOT_IMPLEMENTED_2D = ("Не реализовано в 2D-конвейере (плоскость "
                               "LOS×кросс-рейндж); требует 3D-кинематики "
                               "(ω_eff = ω·cos β).")
        for _lbl, _le in (self.le_omega_axis_y, self.le_omega_axis_z,
                          self.le_precession_rate, self.le_precession_angle):
            _le.setEnabled(False)
            _lbl.setEnabled(False)
            _le.setToolTip((_le.toolTip() or "") + "\n" + _NOT_IMPLEMENTED_2D)
            _lbl.setToolTip((_lbl.toolTip() or "") + "\n" + _NOT_IMPLEMENTED_2D)
        layout.addLayout(model_grid)

        # Параметры обработки
        layout.addWidget(self._section_label("Обработка"))
        proc_layout = QHBoxLayout()
        proc_layout.setSpacing(12)

        self.chk_mocomp = QCheckBox("MOCOMP")
        self.chk_mocomp.setChecked(self.cfg.use_mocomp)
        self.chk_mocomp.stateChanged.connect(self._on_config_changed)
        proc_layout.addWidget(self.chk_mocomp)

        win_lbl, self.cb_window = self._make_combo_widget(
            "Окно (азимут)", WINDOW_NAMES, self.cfg.window,
        )
        winp_lbl, self.le_window_param = self._make_input_widget(
            "Параметр окна (β/at)",
            self.cfg.window_param if self.cfg.window_param is not None else "",
            tooltip="β для kaiser (дефолт 8.0) или at в дБ для chebwin "
                    "(дефолт 60); остальные окна игнорируют",
        )
        disp_lbl, self.cb_display = self._make_combo_widget(
            "Отображение", ["linear", "dB"], self.cfg.display_mode,
        )
        proc_layout.addWidget(win_lbl)
        proc_layout.addWidget(self.cb_window)
        proc_layout.addWidget(winp_lbl)
        proc_layout.addWidget(self.le_window_param)
        proc_layout.addWidget(disp_lbl)
        proc_layout.addWidget(self.cb_display)
        proc_layout.addStretch()
        layout.addLayout(proc_layout)

        # Эффекты реализма
        self._build_realism_section(layout)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ---------- Эффекты реализма ----------

    def _build_realism_section(self, layout):
        """Секция «Реализм»: пресет + чекбоксы/поля эффектов."""
        layout.addWidget(self._section_label("Реализм модели"))

        preset_lbl, self.cb_realism_preset = self._make_combo_widget(
            "Пресет", ["Идеальный", "Реалистичный", "Пользовательский"],
            getattr(self.cfg, "realism_preset", "Идеальный"),
        )
        self.cb_realism_preset.currentTextChanged.connect(self._on_realism_preset)
        preset_row = QHBoxLayout()
        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self.cb_realism_preset)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        eff_grid = QGridLayout()
        eff_grid.setSpacing(4)

        self.chk_complex_sigma = QCheckBox("Случайные фазы рассеивателей")
        self.chk_complex_sigma.setChecked(self.cfg.use_complex_sigma)
        self.chk_complex_sigma.setToolTip(
            "sigma_i * exp(j*phi_i), phi_i ~ U(0, 2*pi).\n"
            "Реальные центры рассеяния имеют случайную фазу — картинка "
            "становится «спекл-подобной» вместо детерминированной."
        )

        self.chk_anisotropy = QCheckBox("Анизотропия рассеяния")
        self.chk_anisotropy.setChecked(self.cfg.use_anisotropy)
        self.chk_anisotropy.setToolTip(
            "Гауссова угловая зависимость sigma_i(theta):\n"
            "отражатель «светится» в пределах ширины лепестка."
        )
        self.le_aniso_width = self._make_input_widget(
            "Ширина лепестка, град", self.cfg.anisotropy_width_deg,
            tooltip="Ширина гауссова лепестка анизотропии (град).",
        )

        self.chk_shadowing = QCheckBox("Затенение рассеивателей")
        self.chk_shadowing.setChecked(self.cfg.use_shadowing)
        self.chk_shadowing.setToolTip(
            "Точка за панелью не видна: гасится, если ближе по линии визия\n"
            "есть другой рассеиватель с поперечным смещением < радиуса."
        )
        self.le_shadow_radius = self._make_input_widget(
            "Радиус затенения, м", self.cfg.shadow_radius_m,
            tooltip="Поперечный радиус «тени» (м).",
        )

        self.chk_antenna = QCheckBox("ДН антенны (двухпутевая)")
        self.chk_antenna.setChecked(self.cfg.use_antenna_pattern)
        self.chk_antenna.setToolTip(
            "Амплитуда эха взвешивается на G^2(theta):\n"
            "гауссова ДН с шириной из поля «Ширина ДН» + ошибка наведения."
        )
        self.le_pointing = self._make_input_widget(
            "Ошибка наведения, град", self.cfg.pointing_error_deg,
            tooltip="Постоянное отклонение луча от цели (град).",
        )
        self.le_drift = self._make_input_widget(
            "Дрейф луча, град", self.cfg.antenna_drift_deg,
            tooltip="Линейный дрейф отклонения за апертуру (град).",
        )

        self.chk_phase_noise = QCheckBox("Фазовый шум генератора")
        self.chk_phase_noise.setChecked(self.cfg.use_phase_noise)
        self.chk_phase_noise.setToolTip(
            "Винеровский фазовый процесс exp(j*phi(t_n)):\n"
            "главный источник расфокусировки на секундных накоплениях."
        )
        self.le_phase_std = self._make_input_widget(
            "RMS фазы, град", self.cfg.phase_noise_std_deg,
            tooltip="RMS фазового набега за апертуру (град). 10 = заметная расфокусировка.",
        )

        self.le_accel = self._make_input_widget(
            "Ускорение, м/с²", self.cfg.accel,
            tooltip="Ускорение цели вдоль направления движения.\n"
                    "Даёт квадратичную фазовую ошибку — классический кейс для автофокуса.",
        )
        self.le_gamma = self._make_input_widget(
            "Угл. ускорение, рад/с²", self.cfg.gamma,
            tooltip="Угловое ускорение вращения, рад/с²; 0 = равномерное.\n"
                    "Маневрирующая цель: theta = omega*t + 0.5*gamma*t^2 (C&M, (1.13)).\n"
                    "При gamma≠0 «слепой» PFA расфокусируется — ожидаемо.",
        )
        self.le_vib_amp = self._make_input_widget(
            "Микровибрация, мм", self.cfg.vib_amplitude_mm,
            tooltip="Амплитуда гармонической микровибрации по дальности (мм).\n"
                    "Создаёт микродоплеровские осцилляции фазы.",
        )
        self.le_vib_freq = self._make_input_widget(
            "Частота вибрации, Гц", self.cfg.vib_frequency_hz,
            tooltip="Частота микровибрации (Гц).",
        )

        eff_grid.addWidget(self.chk_complex_sigma, 0, 0, 1, 2)
        eff_grid.addWidget(self.chk_anisotropy, 1, 0)
        eff_grid.addWidget(self.le_aniso_width[0], 1, 1); eff_grid.addWidget(self.le_aniso_width[1], 1, 2)
        eff_grid.addWidget(self.chk_shadowing, 2, 0)
        eff_grid.addWidget(self.le_shadow_radius[0], 2, 1); eff_grid.addWidget(self.le_shadow_radius[1], 2, 2)
        eff_grid.addWidget(self.chk_antenna, 3, 0)
        eff_grid.addWidget(self.le_pointing[0], 3, 1); eff_grid.addWidget(self.le_pointing[1], 3, 2)
        eff_grid.addWidget(self.le_drift[0], 4, 1); eff_grid.addWidget(self.le_drift[1], 4, 2)
        eff_grid.addWidget(self.chk_phase_noise, 5, 0)
        eff_grid.addWidget(self.le_phase_std[0], 5, 1); eff_grid.addWidget(self.le_phase_std[1], 5, 2)
        eff_grid.addWidget(self.le_accel[0], 6, 0); eff_grid.addWidget(self.le_accel[1], 6, 1)
        eff_grid.addWidget(self.le_gamma[0], 7, 0); eff_grid.addWidget(self.le_gamma[1], 7, 1)
        eff_grid.addWidget(self.le_vib_amp[0], 6, 2); eff_grid.addWidget(self.le_vib_amp[1], 6, 3)
        eff_grid.addWidget(self.le_vib_freq[0], 7, 2); eff_grid.addWidget(self.le_vib_freq[1], 7, 3)
        layout.addLayout(eff_grid)

        # Случайные чекбоксы/поля тоже влияют на конфиг
        for chk in (self.chk_complex_sigma, self.chk_anisotropy,
                    self.chk_shadowing, self.chk_antenna, self.chk_phase_noise):
            chk.stateChanged.connect(self._on_config_changed)

    def _on_realism_preset(self, preset_name):
        """Применить пресет реализма к чекбоксам и полям.

        «Пользовательский» ничего не меняет (значения остаются как есть).
        """
        if preset_name == "Пользовательский":
            self._on_config_changed()
            return
        from config import REALISM_PRESETS
        preset = REALISM_PRESETS.get(preset_name, {})
        if "use_complex_sigma" in preset:
            self.chk_complex_sigma.setChecked(preset["use_complex_sigma"])
        if "use_anisotropy" in preset:
            self.chk_anisotropy.setChecked(preset["use_anisotropy"])
        if "use_shadowing" in preset:
            self.chk_shadowing.setChecked(preset["use_shadowing"])
        if "use_antenna_pattern" in preset:
            self.chk_antenna.setChecked(preset["use_antenna_pattern"])
        if "use_phase_noise" in preset:
            self.chk_phase_noise.setChecked(preset["use_phase_noise"])
        if "accel" in preset:
            self.le_accel[1].setText(str(preset["accel"]))
        if "vib_amplitude_mm" in preset:
            self.le_vib_amp[1].setText(str(preset["vib_amplitude_mm"]))
        self._on_config_changed()

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

        self.btn_help = self._make_button("Справка", self._show_help)
        layout.addWidget(self.btn_help)

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
        # Вкладка 4: Сигнатуры (TF-анализ: STFT/WVD/RID)
        self.tabs.addTab(self._build_tf_tab(), "Сигнатуры")
        # Вкладка 5: Поток кадров
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

        # R3: кросс-масштабирование по данным (chirp-rate)
        self.chk_cross_scale = QCheckBox("Кросс-масштабирование по данным (chirp-rate)")
        self.chk_cross_scale.setChecked(self.cfg.cross_range_scaling == "auto")
        self.chk_cross_scale.setToolTip(
            "Оценка эффективной угловой скорости Omega_eff по линейному тренду "
            "чирпа вдоль кросс-дальности; ось азимута перемасштабируется в метры. "
            "Off = прежний конвейер (битовая совместимость)."
        )
        self.chk_cross_scale.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self.chk_cross_scale)

        return widget

    def _build_tf_tab(self):
        """Вкладка «Сигнатуры»: TF-анализ (STFT/WVD) и RID-куб (§7.6)."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._section_label("Сигнатуры (TF-анализ)"))

        bin_lbl = QLabel("Дальний бин")
        bin_lbl.setObjectName("status_info")
        self.tf_bin_combo = QComboBox()
        self.tf_bin_combo.addItem("Суммарное эхо")
        self.tf_bin_combo.currentIndexChanged.connect(self._on_tf_bin_changed)
        left.addWidget(bin_lbl)
        left.addWidget(self.tf_bin_combo)

        tr_lbl = QLabel("Преобразование")
        tr_lbl.setObjectName("status_info")
        self.tf_transform_combo = QComboBox()
        self.tf_transform_combo.addItems(["STFT", "WVD"])
        self.tf_transform_combo.currentIndexChanged.connect(self._on_tf_transform_changed)
        left.addWidget(tr_lbl)
        left.addWidget(self.tf_transform_combo)

        self.tf_marginal_check = QCheckBox("Маргинальная сумма (≈ Фурье)")
        self.tf_marginal_check.setToolTip(
            "I(r, f) = Σ_t Q(r, f, t) — маргинал RID-куба (TF, (5.18)); "
            "при hop=1 совпадает с азимутальным Фурье-изображением."
        )
        self.tf_marginal_check.stateChanged.connect(self._redraw_tf)
        left.addWidget(self.tf_marginal_check)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)
        slider_row.addWidget(QLabel("Кадр t"))
        self.tf_frame_slider = QSlider(Qt.Horizontal)
        self.tf_frame_slider.setRange(0, 0)
        self.tf_frame_slider.setEnabled(False)
        self.tf_frame_slider.setToolTip("Кадр времени RID-куба Q(r, f, t)")
        self.tf_frame_slider.valueChanged.connect(self._redraw_tf)
        self.tf_frame_label = QLabel("—")
        self.tf_frame_label.setObjectName("status_info")
        slider_row.addWidget(self.tf_frame_slider, stretch=1)
        slider_row.addWidget(self.tf_frame_label)
        left.addLayout(slider_row)

        self.tf_status_label = QLabel(
            "Нет данных. Включите tf_enabled в config для TF-анализа."
        )
        self.tf_status_label.setObjectName("status_info")
        self.tf_status_label.setWordWrap(True)
        left.addWidget(self.tf_status_label)
        left.addStretch()

        right = QVBoxLayout()
        self.pg_tf = pg.PlotWidget(background="#fafafa")
        self.pg_tf.setLabel("left", "Доплер, Гц")
        self.pg_tf.setLabel("bottom", "Время, с")
        self.pg_tf.setTitle("Сигнатуры (STFT)")
        self.tf_img = pg.ImageItem()
        self.pg_tf.getPlotItem().getViewBox().addItem(self.tf_img)
        right.addWidget(self.pg_tf, stretch=1)

        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=3)
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

    def _show_help(self):
        """Открыть справочное окно."""
        dlg = HelpDialog(self)
        dlg.show()

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

    def _make_effects(self):
        """Собрать SimEffects из текущей конфигурации GUI."""
        from simulation.raw_generator import SimEffects
        return SimEffects(
            seed=int(self.cfg.seed) if self.cfg.seed else 42,
            use_complex_sigma=self.cfg.use_complex_sigma,
            use_anisotropy=self.cfg.use_anisotropy,
            anisotropy_width_deg=self.cfg.anisotropy_width_deg,
            use_shadowing=self.cfg.use_shadowing,
            shadow_radius_m=self.cfg.shadow_radius_m,
            use_antenna_pattern=self.cfg.use_antenna_pattern,
            pointing_error_deg=self.cfg.pointing_error_deg,
            antenna_drift_deg=self.cfg.antenna_drift_deg,
            use_phase_noise=self.cfg.use_phase_noise,
            phase_noise_std_deg=self.cfg.phase_noise_std_deg,
        )

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
        self.cfg.pri = max(self._safe_float(self.le_pri[1], self.cfg.pri), 1e-6)
        self.cfg.num_pulses = max(self._safe_int(self.le_num_pulses[1], self.cfg.num_pulses), 1)

        # Модель цели + 3D-кинематика (Tier 2)
        self.cfg.target_model = self.cb_target_model.currentText()
        self.cfg.omega_axis_y = self._safe_float(self.le_omega_axis_y[1], self.cfg.omega_axis_y)
        self.cfg.omega_axis_z = self._safe_float(self.le_omega_axis_z[1], self.cfg.omega_axis_z)
        self.cfg.precession_rate = self._safe_float(self.le_precession_rate[1], self.cfg.precession_rate)
        self.cfg.precession_angle_deg = self._safe_float(self.le_precession_angle[1], self.cfg.precession_angle_deg)

        # Обработка
        self.cfg.use_mocomp = self.chk_mocomp.isChecked()
        self.cfg.window = self.cb_window.currentText()
        self.cfg.window_param = self._safe_float(self.le_window_param, self.cfg.window_param)
        self.cfg.display_mode = self.cb_display.currentText()
        self.cfg.snr_db = self._safe_float(self.le_snr[1], self.cfg.snr_db)

        # R3: кросс-масштабирование по данным (chirp-rate)
        self.cfg.cross_range_scaling = "auto" if self.chk_cross_scale.isChecked() else "off"

        # Реализм
        self.cfg.realism_preset = self.cb_realism_preset.currentText()
        self.cfg.use_complex_sigma = self.chk_complex_sigma.isChecked()
        self.cfg.use_anisotropy = self.chk_anisotropy.isChecked()
        self.cfg.anisotropy_width_deg = self._safe_float(self.le_aniso_width[1], self.cfg.anisotropy_width_deg)
        self.cfg.use_shadowing = self.chk_shadowing.isChecked()
        self.cfg.shadow_radius_m = self._safe_float(self.le_shadow_radius[1], self.cfg.shadow_radius_m)
        self.cfg.use_antenna_pattern = self.chk_antenna.isChecked()
        self.cfg.pointing_error_deg = self._safe_float(self.le_pointing[1], self.cfg.pointing_error_deg)
        self.cfg.antenna_drift_deg = self._safe_float(self.le_drift[1], self.cfg.antenna_drift_deg)
        self.cfg.use_phase_noise = self.chk_phase_noise.isChecked()
        self.cfg.phase_noise_std_deg = self._safe_float(self.le_phase_std[1], self.cfg.phase_noise_std_deg)
        self.cfg.accel = self._safe_float(self.le_accel[1], self.cfg.accel)
        self.cfg.gamma = self._safe_float(self.le_gamma[1], self.cfg.gamma)
        self.cfg.vib_amplitude_mm = self._safe_float(self.le_vib_amp[1], self.cfg.vib_amplitude_mm)
        self.cfg.vib_frequency_hz = self._safe_float(self.le_vib_freq[1], self.cfg.vib_frequency_hz)
        self.cfg.seed = int(self.cfg.seed) if self.cfg.seed else 42

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
            window_param=self.cfg.window_param,
            range_window=self.cfg.range_window,
            display_mode=self.cfg.display_mode,
            snr_db=self.cfg.snr_db,
            effects=self._make_effects(),
            accel=self.cfg.accel,
            gamma=self.cfg.gamma,
            vib_amplitude=self.cfg.vib_amplitude_mm / 1000.0,
            vib_frequency=self.cfg.vib_frequency_hz,
            target_model=self.cfg.target_model,
            satellite=self.cfg.satellite,
            omega_axis_y=self.cfg.omega_axis_y,
            omega_axis_z=self.cfg.omega_axis_z,
            precession_rate=self.cfg.precession_rate,
            precession_angle_deg=self.cfg.precession_angle_deg,
            tf_enabled=self.cfg.tf_enabled,
            tf_n_frames=self.cfg.tf_n_frames,
            tf_window=self.cfg.tf_window,
            cross_range_scaling=self.cfg.cross_range_scaling,
        )

        # Подключаем сигналы промежуточных результатов
        self.processor.range_profiles_ready.connect(self._on_range_profiles)
        self.processor.mocomp_envelope_ready.connect(self._on_mocomp_envelope)
        self.processor.mocomp_phase_ready.connect(self._on_mocomp_phase)
        self.processor.mocomp_before_after.connect(self._on_mocomp_before_after)
        self.processor.isar_ready.connect(self._on_isar_ready)
        self.processor.signal_data_ready.connect(self._on_signal_data)
        self.processor.tf_ready.connect(self._on_tf_ready)
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
        # При включённом keystone профили рассчитаны ПОСЛЕ коррекции (F6:
        # keystone на сырых E до сжатия) — отражаем в подписи вкладки.
        suffix = " (после keystone)" if (getattr(self.processor, "use_keystone",
                                          False) if self.processor is not None else False) else ""
        self.tabs.setTabText(1, f"Профили дальности{suffix} (Δr={dr:.3f} м)")
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

    # ================================================================
    #  Вкладка «Сигнатуры»: tf_ready и отрисовка TF/RID (§7.6)
    # ================================================================

    def _on_tf_ready(self, data):
        """Новые данные TF-анализа: наполнить селекторы и перерисовать."""
        self._tf_data = data
        self._tf_wvd_cache = {}
        top = data.get("top_bins", [])
        self.tf_bin_combo.blockSignals(True)
        self.tf_bin_combo.clear()
        self.tf_bin_combo.addItem("Суммарное эхо")
        for b in top:
            self.tf_bin_combo.addItem(f"Бин {b}")
        self.tf_bin_combo.blockSignals(False)

        n_frames = int(len(data.get("frame_idx", [])))
        self.tf_frame_slider.blockSignals(True)
        self.tf_frame_slider.setRange(0, max(0, n_frames - 1))
        self.tf_frame_slider.setValue(0)
        self.tf_frame_slider.blockSignals(False)

        meta = data.get("meta", {})
        approx = "да" if data.get("approximate") else "нет"
        self.tf_status_label.setText(
            f"RID-куб {tuple(data['Q_mag'].shape)} | кадров {n_frames} | "
            f"nperseg {meta.get('nperseg', '?')} | hop {meta.get('hop', '?')} | "
            f"approximate: {approx}"
        )
        self._redraw_tf()

    def _on_tf_bin_changed(self, *_):
        if hasattr(self, "_tf_data") and self._tf_data is not None:
            self.tf_frame_slider.setValue(0)
            self._redraw_tf()

    def _on_tf_transform_changed(self, *_):
        # WVD для суммарного эха не рассчитываем — подставляем топ-бин (§7.6)
        if self.tf_transform_combo.currentText() == "WVD":
            data = getattr(self, "_tf_data", None)
            if data is not None and self.tf_bin_combo.currentText() == "Суммарное эхо":
                top = data.get("top_bins", [])
                if top:
                    self.tf_bin_combo.blockSignals(True)
                    self.tf_bin_combo.setCurrentText(f"Бин {top[0]}")
                    self.tf_bin_combo.blockSignals(False)
        if hasattr(self, "_tf_data") and self._tf_data is not None:
            self.tf_frame_slider.setValue(0)
            self._redraw_tf()

    def _current_tf_bin(self):
        text = self.tf_bin_combo.currentText()
        if text == "Суммарное эхо":
            return int(self._tf_data["top_bins"][0])
        return int(text.split()[-1])

    def _tf_wvd_image(self, bin_idx):
        """WVD бина (лениво + кэш): строки = время, столбцы = частота."""
        if getattr(self, "_tf_wvd_cache", None) is None:
            self._tf_wvd_cache = {}
        key = str(bin_idx)
        if key not in self._tf_wvd_cache:
            data = self._tf_data
            row = np.asarray(data["rows"][key])
            t_ax, f_ax, W = wigner_ville(row, fs=data["fs"])
            db = 20.0 * np.log10(np.abs(W) + 1e-12)
            db -= float(db.max())
            self._tf_wvd_cache[key] = (
                np.ascontiguousarray(db), np.asarray(t_ax), np.asarray(f_ax)
            )
        return self._tf_wvd_cache[key]

    def _show_tf_image(self, img, x_axis, y_axis, x_label, y_label, title):
        """imshow(S_db) — imshow(..., aspect="auto", vmin=-40, vmax=0)."""
        arr = np.asarray(img, dtype=float)
        self.tf_img.setImage(arr, levels=(-40, 0))
        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        if y_axis is None:
            y_axis = np.arange(arr.shape[0])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])
        self.tf_img.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        vb = self.pg_tf.getPlotItem().getViewBox()
        vb.enableAutoRange()
        vb.updateAutoRange()
        self.pg_tf.setLabel("bottom", x_label)
        self.pg_tf.setLabel("left", y_label)
        self.pg_tf.setTitle(title)

    def _redraw_tf(self, *_):
        data = getattr(self, "_tf_data", None)
        if data is None:
            return
        f = np.asarray(data["f"])
        t = np.asarray(data["t"])
        frame_idx = np.asarray(data["frame_idx"])
        M = np.asarray(data["Q_mag"]).shape[0]
        slider = self.tf_frame_slider.value()

        if self.tf_marginal_check.isChecked():
            # I(r, f) = Σ_t Q — маргинал куба, ≈ азимутальное Фурье (TF, 5.18)
            self.tf_frame_slider.setEnabled(False)
            marg = np.asarray(rid_marginal(data))
            marg_db = 20.0 * np.log10(marg + 1e-12)
            marg_db -= float(marg_db.max())
            self._show_tf_image(
                marg_db, f, np.arange(M),
                "Доплер, Гц", "Дальность, бин",
                "RID-маргинал Σ_t Q (≈ Фурье-изображение)")
            return

        self.tf_frame_slider.setEnabled(True)

        # Слйдер «Кадр t» выбирает мгновенный RID-кадр куба Q(r, f, t)
        if slider != 0:
            k = int(frame_idx[slider])
            self.tf_frame_label.setText(f"Кадр {k} / {t[k]:.3f} с")
            self._show_tf_image(
                rid_frame(data, slider), f, np.arange(M),
                "Доплер, Гц", "Дальность, бин",
                f"RID-кадр t = {t[k]:.3f} с")
            return

        self.tf_frame_label.setText("—")

        if self.tf_transform_combo.currentText() == "WVD":
            bin_idx = self._current_tf_bin()
            img, t_ax, f_ax = self._tf_wvd_image(bin_idx)
            self._show_tf_image(img, f_ax, t_ax,
                                "Доплер, Гц", "Время, с",
                                f"WVD, бин {bin_idx}")
            return

        # STFT-спектрограмма: суммарное эхо или выбранный бин
        if self.tf_bin_combo.currentText() == "Суммарное эхо":
            sig, title = data["total_echo"], "Спектрограмма суммарного эха"
        else:
            bin_idx = self._current_tf_bin()
            sig, title = data["signatures"][str(bin_idx)], f"Спектрограмма бина {bin_idx}"
        self._show_tf_image(
            np.asarray(sig["S_db"]), np.asarray(sig["t"]), np.asarray(sig["f"]),
            "Время, с", "Доплер, Гц", title)

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

        save_dir = os.path.join(app_dir(), "radioimage")

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
            window_param=self.cfg.window_param,
            range_window=self.cfg.range_window,
            display_mode=self.cfg.display_mode,
            snr_db=self.cfg.snr_db,
            effects=self._make_effects(),
            accel=self.cfg.accel,
            gamma=self.cfg.gamma,
            vib_amplitude=self.cfg.vib_amplitude_mm / 1000.0,
            vib_frequency=self.cfg.vib_frequency_hz,
            target_model=self.cfg.target_model,
            satellite=self.cfg.satellite,
            omega_axis_y=self.cfg.omega_axis_y,
            omega_axis_z=self.cfg.omega_axis_z,
            precession_rate=self.cfg.precession_rate,
            precession_angle_deg=self.cfg.precession_angle_deg,
            cross_range_scaling=self.cfg.cross_range_scaling,
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
        save_dir = os.path.join(app_dir(), "radioimage")
        if not os.path.exists(save_dir):
            self.info_label.setText("Нет сохранённых результатов")
            return

        self._frame_index = 0
        self._save_dir = save_dir
        self.info_label.setText("Вывод кадров...")
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._show_next_stream_frame)
        self._render_timer.start(75)
        self.tabs.setCurrentIndex(5)
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
