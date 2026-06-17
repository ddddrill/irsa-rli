import math
import logging
import os
import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
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

    # Логирование в GUI
    progress_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, filename, method, f_c, Xmax, Ymax, spectr_w, ang, range_m,
                 nifft_size=1024, save_dir=None, mode="single",
                 V=0.0, alpha=0.0, omega=None, pri=1.0, num_pulses=NUM_IMAGES,
                 use_mocomp=True, window="none", display_mode="dB", snr_db=float('inf')):
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
        self.snr_db = snr_db
        if omega is not None and omega > 0:
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
        )

    def generate_raw_data(self):
        """Сгенерировать полную матрицу сырых данных SFCW.

        Returns:
            E: комплексная матрица M×N (частоты × импульсы).
            f_r: вектор частот (M,).
            target: объект Target (для доступа к кинематике).
        """
        target = self._make_target()
        E, f_r = generate_raw_matrix(self.radar, target)
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
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m)
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
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m)
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

        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m)

        if use_mocomp:
            P_proc, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
        else:
            P_proc = P.astype(np.complex128)

        I, azimuth_axis, _ = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=window,
        )
        amplitude = compute_amplitude(I, mode="linear")
        return I, amplitude, range_axis, azimuth_axis, target

    def _compute_isar_polar_sync(self, E, f_r, target):
        """Синхронный конвейер с полярным переформатированием (без эмиссии сигналов)."""
        kspace = build_kspace_grid(f_r, self.omega, self.pri, E.shape[1])
        k, theta = kspace["k"], kspace["theta"]
        grid = design_cartesian_grid(k, theta, self.Xmax, self.Ymax)
        E_cart = interpolate_polar_to_cartesian(E, k, theta, grid, method="linear")
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

        if self.method == "с полярным переформатированием":
            return self._compute_isar_polar(E, f_r, target, display_mode)
        else:
            return self._compute_isar_standard(E, f_r, target, display_mode)

    def _compute_isar_standard(self, E, f_r, target, display_mode):
        """Стандартный конвейер: 1D range + 1D azimuth."""
        # 2. Сжатие по дальности
        self.progress_message.emit("Сжатие по дальности...")
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m)
        self.range_profiles_ready.emit(P_abs, range_axis, dr)

        # 3. Компенсация движения (опционально)
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

        # 4. Азимутальное сжатие
        self.progress_message.emit("Азимутальное сжатие...")
        I, azimuth_axis, _ = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=self.window,
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
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=self.range_m)
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
        P, P_abs, range_axis, dr = range_compress(E_partial, f_r, R0=self.range_m)

        if self.use_mocomp:
            P_proc, shifts, ref_bin, phase_errors = mocomp(P, range_axis)
        else:
            P_proc = P.astype(np.complex128)

        I, azimuth_axis, _ = azimuth_compress(
            P_proc, self.omega, self.f_c, pri=self.pri, window=self.window,
        )
        return compute_amplitude(I, mode=self.display_mode)

    def _polar_frame(self, E_partial, f_r, k_full, theta_partial, grid_full):
        """Обработать один кадр полярным методом (k-space → Cartesian → 2D IFFT)."""
        # 1. Сжатие по дальности + MOCOMP
        P, P_abs, range_axis, dr = range_compress(E_partial, f_r, R0=self.range_m)
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
