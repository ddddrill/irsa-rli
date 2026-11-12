import math
import logging
import os
import pickle
import traceback

from simulation.resolution import derive_recon

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix, SimEffects
from simulation.applicability import (
    check_applicability,
    format_report,
    hard_failures,
)
from processing.range_compress import range_compress
from processing.mocomp import mocomp
from processing.azimuth_compress import azimuth_compress, compute_amplitude
from processing.isar_processor import StandardISARProcessor, PolarISARProcessor
from processing.polar_reformat import (
    build_kspace_grid,
    design_cartesian_grid,
    interpolate_polar_to_cartesian,
    process_blind_zones,
    form_image_2d,
    compute_amplitude_db,
    compute_image_entropy,
    should_use_polar_reformat,
)

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 3e8
NUM_IMAGES = 16


class DataProcessor(QThread):
    """Вычислительный поток: формирование тензора РЛИ из данных цели.

    Оркестратор: связывает Радар, Цель и Процессор.

    Сигналы для промежуточных результатов конвейера ISAR:
        range_profiles_ready(P_abs, range_axis, dr)
        mocomp_envelope_ready(shifts, num_pulses)
        mocomp_phase_ready(phase_errors)
        mocomp_before_after(P_abs_before, P_abs_after, range_axis)
        isar_ready(amplitude, range_axis, azimuth_axis)
    """

    frame_progress = pyqtSignal(int)
    frame_saved = pyqtSignal(str, int)

    # Промежуточные результаты нового конвейера
    range_profiles_ready = pyqtSignal(object, object, float)
    mocomp_envelope_ready = pyqtSignal(object, int)
    mocomp_phase_ready = pyqtSignal(object)
    mocomp_before_after = pyqtSignal(object, object, object)
    isar_ready = pyqtSignal(object, object, object)
    signal_data_ready = pyqtSignal(object, object)
    tf_ready = pyqtSignal(object)  # dict RID-куба и сигнатур (вкладка «Сигнатуры»)

    # Прегенерационный лимит памяти матрицы E (RP6): bytes = Nf·N·16.
    # 1.5 ГБ — на 2 порядка выше любого валидного конфига демо (~9 МБ),
    # но отсекает попытки аллокации тибибайтов (V=7800 + T_obs=50 c).
    _E_MEMORY_LIMIT_BYTES = 1_500_000_000

    # Логирование в GUI
    progress_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, filename, method, f_c, Xmax, Ymax, spectr_w, ang, range_m,
                 nifft_size=1024, save_dir=None, mode="single",
                 V=0.0, alpha=0.0, omega=None, pri=1.0, num_pulses=NUM_IMAGES,
                 use_mocomp=True, window="none", window_param=None,
                 range_window="none", display_mode="dB", snr_db=float('inf'),
                 effects=None, accel=0.0, vib_amplitude=0.0, vib_frequency=1.0,
                 gamma=0.0,
                 target_model="точечная", omega_axis_y=0.0, omega_axis_z=0.0,
                 precession_rate=0.0, precession_angle_deg=0.0,
                 tf_enabled=False, tf_n_frames=16, tf_window="hann",
                 cross_range_scaling="off",
                 use_keystone=False,
                 enforce=False,
                 satellite=None,
                 res_range_m=None, res_azimuth_m=None):
        super().__init__()
        self.filename = filename
        self.method = method
        self.f_c = f_c * 1e9
        self.Xmax = Xmax
        self.Ymax = Ymax
        self.spectr_w = spectr_w * 1e9
        self.ang = ang
        self.range_m = range_m
        self.nifft_size = nifft_size
        self.save_dir = save_dir
        self.mode = mode
        self.display_mode = display_mode
        self._is_running = True

        self.V = V
        self.alpha = alpha
        self.pri = pri
        self.num_pulses = num_pulses
        self.use_mocomp = use_mocomp
        self.window = window
        self.window_param = window_param
        self.range_window = range_window
        self.snr_db = snr_db
        # Эффекты реализма: None или SimEffects (дефолты — все выключены)
        self.effects = effects if effects is not None else SimEffects()
        # Нелинейная кинематика цели
        self.accel = accel
        self.vib_amplitude = vib_amplitude
        self.vib_frequency = vib_frequency
        # Угловое ускорение вращения цели (C&M, (1.13)): theta = omega*t + 0.5*gamma*t^2
        self.gamma = gamma
        # Модель цели/ось вращения: принимаются от GUI, в 2D-конвейере не используются
        # (3D-обработка форм — в ветке feature/scatterer-3d, вне R2-R8).
        # F8: поля 3D-кинематики отключены в GUI (тултипы «Не реализовано в
        # 2D-конвейере») — параметры остаются в API для обратной совместимости.
        self.target_model = target_model
        self.satellite = satellite
        self.omega_axis_y = omega_axis_y
        self.omega_axis_z = omega_axis_z
        self.precession_rate = precession_rate
        self.precession_angle_deg = precession_angle_deg
        # TF-анализ (вкладка «Сигнатуры»): выключен → битовая совместимость
        self.tf_enabled = tf_enabled
        self.tf_n_frames = tf_n_frames
        self.tf_window = tf_window
        # R3 кросс-масштабирование по данным: "off" (дефолт, как раньше) / "auto"
        self.cross_range_scaling = cross_range_scaling
        # R6 keystone-перешкалирование: False (дефолт) = битовая идентичность
        self.use_keystone = use_keystone
        # Guard'ы применимости (G1): default False = только сообщения WARN
        # («ОШИБКА КОНФИГУРАЦИИ»/«WARN:» через progress_message), расчёт не
        # блокируется. enforce=True — программный API: hard-нарушения
        # (C1a, C2) прерывают расчёт исключением ValueError.
        self.enforce = enforce

        # ---- RP4: инверсия параметризации (разрешения Δr/Δx) --------------
        # При непустых res_*-kwargs аргументы spectr_w/omega/Xmax/Ymax —
        # производные из derive_recon (план RP §3 RP4): «разрешения»-режим.
        # Существующие позиционные/именованные аргументы прежние (задача
        # RP4.1: бит-совместимость всех существующих вызовов и тестов).
        self._recon_warnings = []
        self._recon_plan = None
        if res_range_m is not None and res_azimuth_m is not None:
            plan = derive_recon(
                f_c_hz=self.f_c, dr_m=res_range_m, da_m=res_azimuth_m,
                pri=pri, num_pulses=num_pulses, R0_m=range_m,
                V=V, alpha=alpha, filename=filename, gamma=gamma)
            self.spectr_w = plan.spectr_w_hz        # Гц: НЕ умножается на 1e9
            self.Xmax = plan.Xmax_m
            self.Ymax = plan.Ymax_m
            self._recon_warnings = list(plan.warnings)
            self._recon_plan = plan
            for w in plan.warnings:
                logger.warning("%s", w)
            self.omega = plan.omega_rad_s           # производная ω (Δx → Δθ)
        elif omega is not None and omega > 0:
            self.omega = omega
        else:
            self.omega = math.radians(self.ang) / (self.num_pulses * self.pri)

        self.radar = Radar(
            c=SPEED_OF_LIGHT,
            f_c=self.f_c,
            Xmax=self.Xmax,
            Ymax=self.Ymax,
            spectr_w=self.spectr_w,
            ph_c=0.0,
            ang=self.ang,
            nifft_size=self.nifft_size,
            snr_db=self.snr_db,
        )
        self.v_cos, self.v_sin, self.v_exp, self.v_complx, self.v_floor = (
            self.radar.vectorize_functions()
        )

    def run(self):
        try:
            if self.mode == "stream":
                self._compute_and_save()
            else:
                self.compute_isar_notified(display_mode=self.display_mode)
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Ошибка в потоке обработки")
            self.error_occurred.emit(f"{type(e).__name__}: {e}")

    def stop(self):
        self._is_running = False

    def _make_target(self):
        return Target(
            self.filename,
            pri=self.pri,
            num_pulses=self.num_pulses,
            R0=self.range_m,
            V=self.V,
            alpha=self.alpha,
            omega=self.omega,
            gamma=self.gamma,
            accel=self.accel,
            vib_amplitude=self.vib_amplitude,
            vib_frequency=self.vib_frequency,
        )

    def generate_raw_data(self):
        """Сгенерировать полную матрицу сырых данных SFCW.

        Returns:
            E: комплексная матрица M×N (частоты × импульсы).
            f_r: вектор частот (M,).
            target: объект Target (для доступа к кинематике).

        Raises:
            ValueError: оценка матрицы E превышает лимит памяти
                (RP6-фикс: вместо неконтролируемого MemoryError —
                понятная ошибка конфигурации с рецептом).
        """
        # Прегенерационный лимит памяти: E — Nf×N комплексных (complex128).
        # Дефолтные конфиги ≤ ~10 МБ; приближение к 1.5 ГБ — уже ошибка
        # конфигурации (реальный случай: V=7800 и T_obs=50 c → Nf=1.87e6).
        est_bytes = self.radar.Nf * self.num_pulses * 16
        if est_bytes > self._E_MEMORY_LIMIT_BYTES:
            gb = est_bytes / 1e9
            msg = (f"ОШИБКА КОНФИГУРАЦИИ: матрица E {self.radar.Nf}×{self.num_pulses} "
                   f"≈ {gb:.1f} ГБ превышает лимит "
                   f"{self._E_MEMORY_LIMIT_BYTES / 1e9:.1f} ГБ. Уменьшите обзор "
                   f"(Δr крупнее → Nf меньше) и/или число импульсов N; "
                   f"при V ≠ 0 — уменьшите T_obs = N·PRI.")
            logger.error(msg)
            self.progress_message.emit(msg)
            raise ValueError(msg)
        target = self._make_target()
        try:
            E, f_r = generate_raw_matrix(self.radar, target, effects=self.effects)
        except pickle.UnpicklingError as e:
            # F7: повреждённые модели (усечённый blob) дают понятное сообщение
            # вместо сырого traceback в глубине конвейера. Исходная причина —
            # в __cause__ (raise ... from e); errors_occurred → GUI-статус.
            msg = (f"Модель {self.filename} повреждена (файл усечён в "
                   f"репозитории); расчёт невозможен. Замените .pkl из "
                   f"исправной копии или выберите другую цель.")
            logger.error("UnpicklingError при загрузке %s: %s",
                         self.filename, e)
            self.progress_message.emit(msg)
            raise ValueError(msg) from e
        except ValueError as e:
            # P2-фикс: генератор сам отказывает в слишком большой
            # broadcast-матрице (Nf×N×I) — сообщение уже содержит рецепт.
            # Дублируем его в progress_message, чтобы GUI показал честный
            # статус ошибки, а не голый ValueError.
            logger.error("Ошибка генерации сырых данных: %s", e)
            self.progress_message.emit(str(e))
            raise
        return E, f_r, target

    def compute_range_profiles(self):
        """Сгенерировать сырые данные и выполнить сжатие по дальности.

        Returns:
            P: матрица профилей дальности M×N (комплексная).
            P_abs: модуль |P| (вещественная).
            range_axis: ось дальностей (м) относительно R0.
            dr: разрешение по дальности (м).
            f_r: вектор частот (M,).
            target: объект Target.
        """
        E, f_r, target = self.generate_raw_data()
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        return P, P_abs, range_axis, dr, f_r, target

    def compute_mocomp(self):
        """Полный конвейер: Raw Data -> Range Compression -> MOCOMP.

        Returns:
            P_comp: компенсированная матрица M×N (комплексная).
            range_axis: ось дальностей (м).
            dr: разрешение по дальности (м).
            shifts: массив сдвигов огибающей (N,).
            ref_bin: индекс опорного бина.
            phase_errors: фазовые ошибки (N,).
            f_r: вектор частот (M,).
            target: объект Target.
        """
        E, f_r, target = self.generate_raw_data()
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
        return P_comp, range_axis, dr, shifts, ref_bin, phase_errors, f_r, target

    def compute_isar(self, use_mocomp=True, window="none"):
        """Полный конвейер ISAR: Raw -> Range -> MOCOMP -> Azimuth -> Image.

        Маршрутизация по self.method аналогична compute_isar_notified.

        Args:
            use_mocomp: True — включить MOCOMP, False — пропустить.
            window: тип оконной функции ("hamming", "hann", "none").

        Returns:
            I: комплексная матрица изображения M×N.
            amplitude: амплитуда |I| (вещественная).
            range_axis: ось дальностей (м).
            azimuth_axis: ось азимута (м).
            target: объект Target.
        """
        E, f_r, target = self.generate_raw_data()

        if self.method == "с полярным переформатированием":
            return self._compute_isar_polar_sync(E, f_r, target)

        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)

        if use_mocomp:
            P_proc, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
        else:
            P_proc = P.astype(np.complex128)

        I, azimuth_axis, _ = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=window,
            window_param=self.window_param,
        )
        amplitude = compute_amplitude(I, mode="linear")
        return I, amplitude, range_axis, azimuth_axis, target

    def _compute_isar_polar_sync(self, E, f_r, target):
        """Синхронный конвейер с полярным переформатированием (без эмиссии сигналов)."""
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        if self.use_mocomp:
            P, _, _, _ = mocomp(P, range_axis)
            E_comp = np.fft.fft(np.fft.ifftshift(P, axes=0), axis=0)
        else:
            E_comp = E

        kspace = build_kspace_grid(f_r, self.omega, self.pri, E_comp.shape[1])
        k, theta = kspace["k"], kspace["theta"]
        grid = design_cartesian_grid(k, theta, self.Xmax, self.Ymax)
        E_cart = interpolate_polar_to_cartesian(E_comp, k, theta, grid, method="linear")
        blind = process_blind_zones(
            E_cart, grid["kx_axis"], grid["ky_axis"],
            window_type="taylor", apply_zero_pad=True, pad_factor=2, nbar=4, sll=-35.0,
        )
        E_final = blind["E_final"]
        I_complex, x_axis, y_axis, x_span, y_span = form_image_2d(
            E_final, grid["dkx"], grid["dky"]
        )
        amplitude = np.abs(I_complex)
        return I_complex, amplitude, x_axis, y_axis, target

    def compute_isar_notified(self, display_mode="dB"):
        """Полный конвейер ISAR с испусканием сигналов на каждом этапе.

        Маршрутизация по self.method:
            "стандартный" -> standard pipeline (1D range + 1D azimuth)
            "с полярным переформатированием" -> polar reformat + 2D IFFT

        Args:
            display_mode: "linear" или "dB" для финального изображения.

        Returns:
            I: комплексная матрица изображения.
            amplitude: амплитуда для отображения.
            range_axis: ось дальностей (м).
            azimuth_axis: ось азимута (м).
            target: объект Target.
        """
        # 1. Генерация сырых данных
        self.progress_message.emit("Генерация сырых данных SFCW...")
        E, f_r, target = self.generate_raw_data()
        self.signal_data_ready.emit(E, f_r)

        # 1a. Guard'ы применимости (G1, C1a/C1b/C2/C3/C4/VIB/WALK): эмпирика
        #     по точной кинематике — без блокировки расчёта (инвариант
        #     совместимости с GUI). Сообщения модуля самодостаточны:
        #     hard (C1a, C2) начинаются с «ОШИБКА КОНФИГУРАЦИИ (C…): …»,
        #     warn — с «WARN (C…): …».
        # RP4: сначала деривационные предупреждения (сигналы коннектятся
        # после конструктора, поэтому эмит отложен в точку расчёта).
        for _w in self._recon_warnings:
            self.progress_message.emit(_w)
        checks = check_applicability(self.radar, target)
        if self.method == "с полярным переформатированием":
            # C4 (порог малых углов) актуален только для стандартного RD-
            # тракта; полярная переформатизация большое Δθ как раз и
            # обрабатывает (семантика прежнего диагностического блока).
            checks = [c for c in checks if c.code != "C4"]
        for c in checks:
            if c.ok or math.isinf(c.margin):
                continue
            self.progress_message.emit(c.message)
            logger.warning("%s", c.message)
        if self.enforce:
            failed = hard_failures(checks)
            if failed:
                raise ValueError(
                    "Невалидная конфигурация (enforce=True): "
                    + "; ".join(c.message for c in failed))

        if self.method == "с полярным переформатированием":
            return self._compute_isar_polar(E, f_r, target, display_mode)

        return self._compute_isar_standard(E, f_r, target, display_mode)

    def azimuth_compress(self, P_proc, window="none"):
        """Помощник: спектральное сжатие по азимуту + шкалирование осей.

        Returns:
            I: комплексное изображение M×N.
            azimuth_axis: ось азимута (м).
            doppler_axis: доплеровская ось (Гц), использованная для
                формирования изображения.
        """
        I, azimuth_axis, doppler_axis = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=window,
            window_param=self.window_param,
        )
        return I, azimuth_axis, doppler_axis

    def _maybe_cross_range_scale(self, I, range_axis, doppler_axis, azimuth_axis_cfg):
        """R3: кросс-масштабирование по данным (chirp-rate) при auto.

        Метод "phasefit" выбран как основной для разрежённых сцен после
        автофокусировки: устойчив к низкому С/Ш строк и к смещению
        опорного рассеивателя (перехватывается точкой пересечения b).
        При fallback возвращается конфигурационная ось азимута.
        """
        from processing.cross_range_scaling import cross_range_scale

        self.progress_message.emit("Кросс-масштабирование по данным (chirp-rate)...")
        fs = 1.0 / self.pri
        cr = cross_range_scale(
            I, range_axis, doppler_axis, self.f_c, fs, method="phasefit",
        )
        if cr["fallback"]:
            self.progress_message.emit(
                "Кросс-масштабирование: оценка недостоверна, "
                f"N = {cr['n_used']} точек [fallback → cfg-ось]"
            )
            return azimuth_axis_cfg
        omega_eff = cr["omega_eff"]
        self.progress_message.emit(
            f"Кросс-масштабирование: Ω̂ = {omega_eff:.5f} рад/с "
            f"(истинное ω = {self.omega:.5f}), R² = {cr['r2']:.2f}, "
            f"N = {cr['n_used']} точек"
        )
        return cr["azimuth_axis_m"]

    def _compute_isar_standard(self, E, f_r, target, display_mode):
        """Стандартный конвейер: 1D range + 1D azimuth."""
        if self.pri <= 0:
            raise ValueError(
                f"PRI = {self.pri} с должно быть > 0 для азимутального сжатия"
            )
        # 3a. Keystone-коррекция вращательной миграции ячеек (R6→F6, C&M §5.5):
        #     ПЕРВЫМ, на сырых E — нетронутая f×t-структура фазы
        #     (t_ks = (f_c/f_r)·t). Только по флагу (дефолт off = битовая
        #     идентичность пути). Измерение (План фиксов §6.2, стенды
        #     R/E2/T/K): это единственный порядок, дающий выигрыш —
        #     прежний гибрид «после MOCOMP» был нейтрален (×0.99) и
        #     разрушал per-импульсные фазы, «запечённые» MOCOMP/выравниванием
        #     в E-область (нелинейная (m,j)-фаза); порядок
        #     «envelope → keystone → PPP» измеренно ВРЕДЕН (×0.19–0.56).
        if self.use_keystone:
            from processing.keystone import keystone_transform
            self.progress_message.emit("Keystone-коррекция миграции ячеек...")
            E = keystone_transform(E, f_r, self.f_c)

        # 2. Сжатие по дальности — ЕДИНСТВЕННЫЙ проход: дерамп R0 (per-
        #     строчная константа, keystone её не трогает) и окно применяются
        #     один раз (двойное окно/сжатие прежнего гибрида устранены).
        self.progress_message.emit("Сжатие по дальности...")
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        self.range_profiles_ready.emit(P_abs, range_axis, dr)

        # 3. Компенсация движения (опционально) — после keystone:
        #     per-столбцовые операции в P-области не конфликтуют с
        #     ресемплингом медленного времени.
        if self.use_mocomp:
            self.progress_message.emit("Компенсация движения...")
            P_proc, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
            self.mocomp_envelope_ready.emit(shifts, self.num_pulses)
            self.mocomp_phase_ready.emit(phase_errors)
            self.mocomp_before_after.emit(P_abs, np.abs(P_proc), range_axis)
        else:
            P_proc = P.astype(np.complex128)
            self.mocomp_envelope_ready.emit(np.zeros(self.num_pulses), self.num_pulses)
            self.mocomp_phase_ready.emit(np.zeros(self.num_pulses))
            self.mocomp_before_after.emit(P_abs, P_abs, range_axis)

        # 3b. TF-анализ (вкладка «Сигнатуры»): RID-куб и спектрограммы топ-K
        #     бинов — только по включённому флагу (дефолт off = без изменений).
        #     P_proc уже выровнен MOCOMP-блоком; сигнал эмитится один раз.
        if self.tf_enabled:
            from processing.tf_analysis import (
                micro_doppler_signature,
                rid_cube,
                strongest_range_bins,
                total_echo_signature,
            )
            self.progress_message.emit("TF-анализ: STFT…")
            fs = 1.0 / self.pri
            top_k = strongest_range_bins(P_proc, k=5)
            rid = rid_cube(P_proc, fs=fs, n_frames=self.tf_n_frames,
                           window=self.tf_window)
            signatures = {}
            for b in top_k:
                f_b, t_b, s_db = micro_doppler_signature(
                    P_proc, b, fs=fs, window=self.tf_window)
                signatures[str(int(b))] = {"f": f_b, "t": t_b, "S_db": s_db}
            total_echo = total_echo_signature(P_proc, fs=fs,
                                              window=self.tf_window)
            rows = {str(int(b)): np.asarray(P_proc[b]) for b in top_k}
            self.tf_ready.emit({
                "Q_mag": rid["Q_mag"],
                "f": rid["f"],
                "t": rid["t"],
                "frame_idx": rid["frame_idx"],
                "approximate": rid["approximate"],
                "meta": rid["meta"],
                "top_bins": [int(b) for b in top_k],
                "signatures": signatures,
                "total_echo": total_echo,
                "rows": rows,
                "fs": fs,
                "n_frames": self.tf_n_frames,
            })
            self.progress_message.emit(
                f"TF-анализ: RID-куб {rid['Q_mag'].shape} готов"
            )

        # 4. Азимутальное сжатие
        self.progress_message.emit("Азимутальное сжатие...")
        I, azimuth_axis, doppler_axis = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=self.window,
            window_param=self.window_param,
        )
        # 4b. Кросс-масштабирование по данным (R3): только при "auto",
        #     off = битовая совместимость с прежним конвейером.
        if self.cross_range_scaling == "auto":
            azimuth_axis = self._maybe_cross_range_scale(
                I, range_axis, doppler_axis, azimuth_axis,
            )
        amplitude = compute_amplitude(I, mode=display_mode)
        self.isar_ready.emit(amplitude, range_axis, azimuth_axis)

        entropy = compute_image_entropy(I)
        peak = np.max(np.abs(I))
        self.progress_message.emit(
            f"Готово | Размер: {I.shape[0]}x{I.shape[1]} | "
            f"Энтропия: {entropy:.2f} | Пик: {peak:.1f}"
        )

        return I, amplitude, range_axis, azimuth_axis, target

    def _compute_isar_polar(self, E, f_r, target, display_mode):
        """Конвейер с полярным переформатированием (через модуль polar_reformat).

        Полный pipeline:
            1. Сжатие по дальности (IFFT) — для MOCOMP
            2. MOCOMP — компенсация поступательного движения
            3. Обратное сжатие (FFT) — возврат в k-space с компенсацией
            4. Полярная сетка k-space — build_kspace_grid
            5. Декартова сетка — design_cartesian_grid
            6. Интерполяция (k,theta) -> (kx,ky) — interpolate_polar_to_cartesian
            7. Слепые зоны + Taylor window + zero padding — process_blind_zones
            8. 2D IFFT + физическое масштабирование — form_image_2d
        """
        if self.omega == 0:
            logger.warning(
                "Полярное переформатирование невозможно при omega=0, "
                "используется стандартный метод"
            )
            return self._compute_isar_standard(E, f_r, target, display_mode)

        # 1. Сигналы для вкладки «Сигнал»
        self.signal_data_ready.emit(E, f_r)

        # 2. Сжатие по дальности (нужно для MOCOMP)
        self.progress_message.emit("Сжатие по дальности...")
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        self.range_profiles_ready.emit(P_abs, range_axis, dr)

        # 3. Компенсация движения (опционально)
        if self.use_mocomp:
            self.progress_message.emit("Компенсация движения...")
            P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
            self.mocomp_envelope_ready.emit(shifts, self.num_pulses)
            self.mocomp_phase_ready.emit(phase_errors)
            self.mocomp_before_after.emit(P_abs, np.abs(P_comp), range_axis)
            # 4. Возврат в k-space (частотную область) после компенсации
            #    ifftshift нужен т.к. P_comp был fftshift'нут (zero-range по центру)
            #    Формула: E = fft(ifftshift(P)) — точное обращение P = fftshift(ifft(E))
            E_comp = np.fft.fft(np.fft.ifftshift(P_comp, axes=0), axis=0)
        else:
            self.mocomp_envelope_ready.emit(np.zeros(self.num_pulses), self.num_pulses)
            self.mocomp_phase_ready.emit(np.zeros(self.num_pulses))
            self.mocomp_before_after.emit(P_abs, P_abs, range_axis)
            E_comp = E

        # 5. Полярная сетка k-space: k = 4*pi*f/c, theta = omega*n*pri
        self.progress_message.emit("Построение полярной сетки k-space...")
        kspace = build_kspace_grid(f_r, self.omega, self.pri, E_comp.shape[1])
        k, theta = kspace["k"], kspace["theta"]

        # 6. Декартова сетка с шагом Котельникова: dkx=2*pi/Xmax, dky=2*pi/Ymax
        grid = design_cartesian_grid(k, theta, self.Xmax, self.Ymax)

        # 7. Интерполяция (полярная -> декартова), линейная по 2D
        self.progress_message.emit("Интерполяция на декартову сетку...")
        E_cart = interpolate_polar_to_cartesian(E_comp, k, theta, grid, method="linear")

        # 8. Слепые зоны + Taylor window (nbar=4, SLL=-35dB) + zero pad x2
        self.progress_message.emit("Обработка слепых зон + окно Тейлора...")
        blind = process_blind_zones(
            E_cart, grid["kx_axis"], grid["ky_axis"],
            window_type="taylor", apply_zero_pad=True, pad_factor=2,
            nbar=4, sll=-35.0,
        )
        E_final = blind["E_final"]

        # 9. 2D IFFT + правильное масштабирование осей (2*pi/dkx)
        self.progress_message.emit("2D IFFT + масштабирование...")
        I_complex, x_axis, y_axis, x_span, y_span = form_image_2d(
            E_final, grid["dkx"], grid["dky"]
        )

        # 10. Амплитуда в дБ (или линейная)
        if display_mode == "dB":
            amplitude = compute_amplitude_db(I_complex)
        else:
            amplitude = np.abs(I_complex)

        # isar_ready: I_complex имеет форму (M, N) = (range_bins, cross_range_bins)
        # x_axis — дальность (м), y_axis — азимут (м)
        self.isar_ready.emit(amplitude, x_axis, y_axis)

        entropy = compute_image_entropy(I_complex)
        peak = np.max(np.abs(I_complex))
        self.progress_message.emit(
            f"Готово | Размер: {I_complex.shape[0]}x{I_complex.shape[1]} | "
            f"Энтропия: {entropy:.2f} | Пик: {peak:.1f}"
        )

        return I_complex, amplitude, x_axis, y_axis, target

    def compute_single(self):
        """Вычислить РЛИ для первого импульса синхронно (без потока)."""
        E, f_r, target = self.generate_raw_data()
        Nf, Nph, _, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )

        Es_single = E[:, 0:1]
        base_img, base_x, base_y = proc.compute_image(Es_single)
        return E, f_r, ph_r, base_img, base_x, base_y

    def _compute_and_save(self):
        # RP4: деривационные предупреждения (см. compute_isar_notified)
        for _w in self._recon_warnings:
            self.progress_message.emit(_w)
        if self.save_dir and os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                item_path = os.path.join(self.save_dir, item)
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)

        E, f_r, target = self.generate_raw_data()
        use_polar = (
            self.method == "с полярным переформатированием"
            and self.omega > 0
        )

        # Для полярного метода: предварительно строим полную сетку
        # (на основе полного E с N импульсами), чтобы оси были фиксированными
        if use_polar:
            kspace = build_kspace_grid(f_r, self.omega, self.pri, E.shape[1])
            k_full, theta_full = kspace["k"], kspace["theta"]
            grid_full = design_cartesian_grid(k_full, theta_full, self.Xmax, self.Ymax)

        for i in range(1, self.num_pulses + 1):
            if not self._is_running:
                break

            E_partial = E[:, 0:i]

            if use_polar:
                amplitude = self._polar_frame(
                    E_partial, f_r, k_full, theta_full[:i], grid_full,
                )
            else:
                amplitude = self._standard_frame(E_partial, f_r)

            if self.save_dir:
                self._save_frame(i, amplitude)

            self.frame_progress.emit(i)
            logger.info("Обработан кадр %d/%d", i, self.num_pulses)

    def _standard_frame(self, E_partial, f_r):
        """Обработать один кадр стандартным методом (1D + 1D)."""
        # Keystone (R6→F6, C&M §5.5) — ПЕРВЫМ, на сырых E: нетронутая f×t-
        # структура фазы (см. _compute_isar_standard §6.3 и измерение плана
        # §6.2: порядки «после MOCOMP» / «envelope → keystone» вредны).
        if self.use_keystone:
            from processing.keystone import keystone_transform
            self.progress_message.emit("Keystone-коррекция миграции ячеек...")
            E_partial = keystone_transform(E_partial, f_r, self.f_c)

        P, P_abs, range_axis, dr = range_compress(E_partial, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)

        if self.use_mocomp:
            P_proc, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
        else:
            P_proc = P.astype(np.complex128)

        I, azimuth_axis, _ = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=self.window,
            window_param=self.window_param,
        )
        return compute_amplitude(I, mode=self.display_mode)

    def _polar_frame(self, E_partial, f_r, k_full, theta_partial, grid_full):
        """Обработать один кадр полярным методом (k-space → Cartesian → 2D IFFT)."""
        # 1. Сжатие по дальности + MOCOMP
        P, P_abs, range_axis, dr = range_compress(E_partial, f_r, R0=self.range_m, window=self.range_window,
                           window_param=self.window_param)
        if self.use_mocomp:
            P_comp, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
            E_comp = np.fft.fft(np.fft.ifftshift(P_comp, axes=0), axis=0)
        else:
            E_comp = E_partial

        # 2. Интерполяция на фиксированную декартову сетку
        E_cart = interpolate_polar_to_cartesian(
            E_comp, k_full, theta_partial, grid_full, method="linear",
        )

        # 3. Слепые зоны + окно + zero-pad
        blind = process_blind_zones(
            E_cart, grid_full["kx_axis"], grid_full["ky_axis"],
            window_type="taylor", apply_zero_pad=True, pad_factor=2,
            nbar=4, sll=-35.0,
        )

        # 4. 2D IFFT
        I_complex, x_axis, y_axis, x_span, y_span = form_image_2d(
            blind["E_final"], grid_full["dkx"], grid_full["dky"],
        )

        # 5. Амплитуда
        if self.display_mode == "dB":
            return compute_amplitude_db(I_complex)
        return np.abs(I_complex)

    def _save_frame(self, frame_num, amplitude):
        frame_dir = os.path.join(self.save_dir, f"frame_{frame_num:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        from PIL import Image

        img_data = np.abs(np.rot90(amplitude, 2))
        max_val = img_data.max()
        if max_val > 0:
            img_norm = (img_data / max_val * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_data, dtype=np.uint8)
        Image.fromarray(img_norm).save(os.path.join(frame_dir, "rli.png"))

    def _radioimage_from_ranges(self, intens, ranges):
        """Сформировать РЛИ из вектора дальностей (один импульс).

        Использует формулу: phi = -4*pi*f/c * R_i
        """
        Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )

        ranges_matrix = ranges.reshape(1, -1)
        Es = proc.compute_field_from_ranges(intens, ranges_matrix)
        base_img, base_x, base_y = proc.compute_image(Es)
        return Es, f_r, ph_r, base_img, base_x, base_y

    def radioimage_single(self, intens, x_coord, y_coord):
        if self.method == "стандартный":
            return self._standard_process(intens, x_coord, y_coord)
        elif self.method == "с полярным переформатированием":
            return self._polar_process(intens, x_coord, y_coord)

    def _standard_process(self, intens, x_coord, y_coord):
        Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )

        Es = proc.compute_field(intens=intens, x=x_coord, y=y_coord)
        base_img, base_x, base_y = proc.compute_image(Es)
        return Es, f_r, ph_r, base_img, base_x, base_y

    def _polar_process(self, intens, x_coord, y_coord):
        Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        base_proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )
        Es = base_proc.compute_field(intens=intens, x=x_coord, y=y_coord)

        Nf, Nph, kx, ky, kxMax, kyMax, kxMin, kyMin, Nifft_fr, Nifft_ph, M = (
            self.radar.polar_img_params()
        )

        polar_proc = PolarISARProcessor(
            Nf=Nf, Nph=Nph,
            kx=kx, ky=ky,
            kxMax=kxMax, kyMax=kyMax, kxMin=kxMin, kyMin=kyMin,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph, M=M,
        )

        Es_pol, grid_x, grid_y = polar_proc.polar_reformat(Es)
        img_polar, Kx, Ky, len_xp, len_yp = polar_proc.compute_image(
            Es_pol, grid_x, grid_y
        )
        return Es_pol, grid_x, grid_y, img_polar, Kx, Ky, len_xp, len_yp
