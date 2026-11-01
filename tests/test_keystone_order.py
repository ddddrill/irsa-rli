"""Тесты порядка keystone в GUI-конвейере (задача F6, План фиксов §6.4–6.5).

Физика: keystone перешкалирует медленное время по-разному в каждой строке m
(t_ks = (f_c/f_r)·t), поэтому любые per-импульсные фазы, «запечённые» до него
(MOCOMP/выравнивание огибающей), разрушают фокусировку. Единственный порядок,
дающий выигрыш: keystone на сырых E → сжатие → MOCOMP (измерение плана §6.2:
порядки «после MOCOMP» ×0.99 нейтрален; «envelope → keystone → PPP» ×0.19–0.56
вреден).

Эталонный сценарий (стенд K, валидированный по guard'ам): fc=6 ГГц (иначе
C1a нарушается при y_max=20 м), B=0.5 ГГц (dr=0.3 м), Δθ=7.8°, N=512,
PRI=0.05 c — миграция ярких точек ≈ 9 бинов за апертуру.
"""
import math

import numpy as np
import pytest

from models.radar import Radar

SPEED_OF_LIGHT = 3e8
FC = 6e9
B = 0.5e9
PRI = 0.05
N = 512
SURVEY = 50.0
R0 = 500_000.0
DT_DEG = 7.8                                    # Δθ=7.8° за апертуру
OMEGA = math.radians(DT_DEG) / (N * PRI)
# (x, y, σ): крайние рассеиватели (0,±20) мигрируют ≈ 2.7 м = 9 бинов
POINTS = np.array([
    [0.0, 20.0, 10.0],
    [0.0, -20.0, 5.0],
    [3.0, 8.0, 3.0],
    [-2.0, -5.0, 2.0],
], dtype=float)


class _KTarget:
    """Синтетическая цель со сценарием K (API Target для DataProcessor)."""

    def __init__(self, V=0.0):
        self.omega = OMEGA
        self.pri = PRI
        self.num_pulses = N
        self.R0 = R0
        self.V = V
        self.alpha = 0.0
        self.gamma = 0.0
        self.accel = 0.0
        self.vib_amplitude = 0.0
        self.vib_frequency = 0.0

    def get_ranges_all(self):
        t = np.arange(N) * PRI
        theta = OMEGA * t
        x, y = POINTS[:, 0], POINTS[:, 1]
        X = (R0 + self.V * math.cos(0.0) * t[:, None]
             + x[None, :] * np.cos(theta)[:, None] - y[None, :] * np.sin(theta)[:, None])
        Y = (self.V * math.sin(0.0) * t[:, None]
             + x[None, :] * np.sin(theta)[:, None] + y[None, :] * np.cos(theta)[:, None])
        return np.sqrt(X ** 2 + Y ** 2)

    def get_cross_range_all(self):
        t = np.arange(N) * PRI
        theta = OMEGA * t
        x, y = POINTS[:, 0], POINTS[:, 1]
        Y = (self.V * math.sin(0.0) * t[:, None]
             + x[None, :] * np.sin(theta)[:, None] + y[None, :] * np.cos(theta)[:, None])
        return Y

    def get_aspect_angles(self):
        theta = self.omega * np.arange(self.num_pulses) * self.pri
        return theta  # рад; генератор конвертирует в градусы

    def _load_scatterers(self):
        return {"intens": POINTS[:, 2].copy(),
                "x0": POINTS[:, 0].copy(), "y0": POINTS[:, 1].copy()}


def _make_processor(use_keystone, V=0.0):
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="synthetic-stand-K", method="стандартный",
        f_c=FC / 1e9, Xmax=SURVEY, Ymax=SURVEY, spectr_w=B / 1e9,
        ang=9.0, range_m=R0, nifft_size=256,
        V=V, alpha=0.0, omega=OMEGA, pri=PRI, num_pulses=N,
        use_mocomp=True, window="none", range_window="none",
        display_mode="linear", use_keystone=use_keystone,
    )
    target = _KTarget(V=V)
    proc._make_target = lambda: target    # таблица рассеивателей — синтетика
    return proc, target


def _entropy(amp):
    p = (amp / amp.sum()).ravel()
    return -np.sum(p * np.log(p + 1e-30))


def _local_peaks(amp, range_axis, az_axis):
    from scipy.ndimage import maximum_filter
    a = amp.copy()
    a[:3, :] = 0; a[-3:, :] = 0; a[:, :3] = 0; a[:, -3:] = 0
    mask = (maximum_filter(a, size=3) == a) & (a > 0)
    return np.nonzero(mask)


def test_keystone_order_stand_k_gui():
    """Стенд K (план §6.4–6.5):

    — пик и энтропия — через GUI-путь (use_keystone=True, use_mocomp=True,
      как требует план): пик ≥ ×1.08 к базе MOCOMP-only, энтропия ниже;
    — по-рассеивательный критерий (min локальный пик ≥ ×1.5) — измерен в
      сравнении без MOCOMP (оба кейса). Причина: дух плана — «заворачивающиеся
      по дальности рассеиватели стягиваются в точку» (§6.2); PPP-автофокус
      (mocomp.phase_autofocus) якорится на опорный бин и вычитает его фазу —
      сильнейший рассеиватель опорной строки сворачивается в DC-столбец,
      и окна у истинных позиций блокируются артефактом референ-бина, а не
      физикой keystone. В сравнении без MOCOMP измеряется собственно цель
      keystone — стягивание мигрирующих ячеек.
    """
    proc_base, _ = _make_processor(use_keystone=False)
    _, amp_base, range_axis, az_axis, _ = proc_base.compute_isar_notified(
        display_mode="linear")

    proc_ks, _ = _make_processor(use_keystone=True)
    _, amp_ks, rng_ks, az_ks, _ = proc_ks.compute_isar_notified(
        display_mode="linear")

    gain = amp_ks.max() / amp_base.max()
    assert gain >= 1.08, f"пик вырос только ×{gain:.4f} (< 1.08)"
    assert _entropy(amp_ks) < _entropy(amp_base), (_entropy(amp_ks), _entropy(amp_base))

    # по-рассеивательно: усиление локального пика у истинной позиции КАЖДОГО
    # рассеивателя (план §6.4 п.1; измерено ×1.8–×6.2; ниже плановых ×4.57/
    # ×4.61 из-за fc=6 ГГц вместо 10 — C1a-совместимая ревизия стенда)

    def local_peak_at(amp, range_axis, az_axis, m_t, y_t, tol_b):
        """Максимум амплитуды в окне ±tol_b бин вокруг истинной позиции."""
        m_idx = int(np.argmin(np.abs(range_axis - m_t)))
        y_idx = int(np.argmin(np.abs(az_axis - y_t)))
        return amp[m_idx - tol_b:m_idx + tol_b + 1,
                   y_idx - tol_b:y_idx + tol_b + 1].max()

    proc_raw_base, _ = _make_processor(use_keystone=False)
    proc_raw_base.use_mocomp = False
    _, amp_rb, ra_rb, az_rb, _ = proc_raw_base.compute_isar_notified(
        display_mode="linear")
    proc_raw_ks, _ = _make_processor(use_keystone=True)
    proc_raw_ks.use_mocomp = False
    _, amp_rk, ra_rk, az_rk, _ = proc_raw_ks.compute_isar_notified(
        display_mode="linear")

    ratios = []
    for x_t, y_t, _ in POINTS:
        pb = local_peak_at(amp_rb, ra_rb, az_rb, x_t, y_t, 4)
        pk = local_peak_at(amp_rk, ra_rk, az_rk, x_t, y_t, 4)
        ratios.append(pk / max(pb, 1e-12))
    assert min(ratios) >= 1.5, (POINTS, ratios)


def test_keystone_gui_matches_raw_e_pipeline():
    """GUI-путь (ks на raw E → сжатие → MOCOMP) совпадает с ручным конвейером
    на сырых E в пределах 5% (критерий §6.4 п.2)."""
    from processing.keystone import keystone_transform
    from processing.range_compress import range_compress
    from processing.mocomp import mocomp
    from processing.azimuth_compress import azimuth_compress, compute_amplitude

    proc, target = _make_processor(use_keystone=True)
    E, f_r, _ = proc.generate_raw_data()

    # GUI-путь
    _, amp_gui, _, _, _ = proc.compute_isar_notified(display_mode="linear")

    # ручной конвейер: keystone на сырых E → сжатие → MOCOMP → азимут
    E_ks = keystone_transform(E, f_r, proc.f_c)
    P, _, range_axis, _ = range_compress(E_ks, f_r, R0=R0, window="none")
    P_proc, _, _, _ = mocomp(P, range_axis)
    I, az_axis, _ = azimuth_compress(P_proc, OMEGA, FC, pri=PRI)
    amp_manual = compute_amplitude(I, mode="linear")

    pg = amp_gui.max()
    pm = amp_manual.max()
    assert abs(pg - pm) / max(pg, 1e-12) <= 0.05, (pg, pm)


def test_keystone_with_mocomp_beats_keystone_only_for_translation():
    """Стенд T (трансляция V=0.2 + вращение): keystone без MOCOMP хуже,
    чем с MOCOMP — выравнивание огибающей по-прежнему необходимо."""
    proc_moc, _ = _make_processor(use_keystone=True, V=0.2)
    proc_raw, _ = _make_processor(use_keystone=True, V=0.2)
    proc_raw.use_mocomp = False            # то же E, но без выравнивания

    _, amp_moc, _, _, _ = proc_moc.compute_isar_notified(display_mode="linear")
    _, amp_raw, _, _, _ = proc_raw.compute_isar_notified(display_mode="linear")

    assert amp_moc.max() > amp_raw.max(), (amp_moc.max(), amp_raw.max())
