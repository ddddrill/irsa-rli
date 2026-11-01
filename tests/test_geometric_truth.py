"""V1: тесты геометрической истинности (задача V1, План фиксов §11).

78 старых тестов проверяют энтропию/энергию — ни один не проверял главное
свойство: позиции рассеивателей на изображении против истины. Энтропия не
отличает алиасинг от фокуса. Здесь — только truth-проверки.

Допуск: 2·PSF по каждой оси (PSF_x = 0.886·c/2B, PSF_y = 0.886·λ/(2Δθ), C&M §6.3).
Матчинг: локальные максимумы (scipy.ndimage.maximum_filter) → ближайший к истине.

Тест №4 — canary: намеренно алиасящая конфигурация обязана давать ЛОЖНЫЕ
позиции и репорт guard'а C1a; гарантия «зубастости» тестов 1–3.
"""
import math

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from models.radar import Radar
from models.target import Target
from simulation.applicability import check_applicability
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude
from processing.polar_reformat import (
    build_kspace_grid,
    design_cartesian_grid,
    interpolate_polar_to_cartesian,
    process_blind_zones,
    form_image_2d,
)

SPEED_OF_LIGHT = 3e8
MATRIX = "matrices/matrices_solarB_high.pkl"


# ---------------------------------------------------------------- утилиты ---

def _E_from_points(f_r, omega, pri, N, points, R0):
    """Синтетическая E-матрица точечных рассеивателей (точная геометрия):
    R_i(t) = sqrt((R0 + x cosθ − y sinθ)² + (x sinθ + y cosθ)²), θ = ωt."""
    t = np.arange(N) * pri
    theta = omega * t                        # (N,)
    sx, sy = points[:, 0], points[:, 1]
    X = R0 + sx[None, :] * np.cos(theta)[:, None] - sy[None, :] * np.sin(theta)[:, None]
    Y = sx[None, :] * np.sin(theta)[:, None] + sy[None, :] * np.cos(theta)[:, None]
    R = np.sqrt(X ** 2 + Y ** 2)             # (N, I)
    phi = -4.0 * math.pi * f_r[:, None, None] / SPEED_OF_LIGHT * R[None, :, :]
    return np.exp(1j * phi).sum(axis=2)      # (M, N)


def _local_peak_positions(amp, axis0, axis1, exclude_edge=3):
    """Позиции локальных максимумов в метрах ((m, y) кортежи)."""
    a = amp.copy()
    a[:exclude_edge, :] = 0; a[-exclude_edge:, :] = 0
    a[:, :exclude_edge] = 0; a[:, -exclude_edge:] = 0
    mask = (maximum_filter(a, size=3) == a) & (a > 0)
    rows, cols = np.nonzero(mask)
    return list(zip(axis0[rows], axis1[cols]))


def _nearest_peak(m_true, y_true, peaks):
    """Ближайший (по эвклиду) локальный пик к истинной позиции."""
    dists = [math.hypot(m - m_true, y - y_true) for m, y in peaks]
    k = int(np.argmin(dists))
    return peaks[k]


# ---------------------------------------- 1) стандартный тракт, синтетика ---

class _SyntheticTarget:
    """Лёгкий shim API Target для guard'ов на синтетических точках."""

    def __init__(self, omega, pri, num_pulses, R0, x0, y0):
        self.omega = omega
        self.pri = pri
        self.num_pulses = num_pulses
        self.R0 = R0
        self.V = 0.0
        self.alpha = 0.0
        self.gamma = 0.0
        self.accel = 0.0
        self.vib_amplitude = 0.0
        self.vib_frequency = 0.0
        self._x0, self._y0 = np.asarray(x0), np.asarray(y0)

    def get_ranges_all(self):
        t = np.arange(self.num_pulses) * self.pri
        theta = self.omega * t
        X = (self.R0 + self._x0[None, :] * np.cos(theta)[:, None]
             - self._y0[None, :] * np.sin(theta)[:, None])
        Y = (self._x0[None, :] * np.sin(theta)[:, None]
             + self._y0[None, :] * np.cos(theta)[:, None])
        return np.sqrt(X ** 2 + Y ** 2)

    def _load_scatterers(self):
        return {"intens": np.ones(self._x0.shape[0]),
                "x0": self._x0, "y0": self._y0}


@pytest.mark.parametrize("Xmax", [30.0, 40.0])   # окно ≥ 15.5 м: ±8-м точки с запасом C2 ≥ 1.9
def test_standard_truth_5points(Xmax):
    """5 точечных рассеивателей, Δθ=1°: позиции на оси = истины."""
    fc, B = 8e9, 0.5e9
    pri, N = 0.05, 256
    omega = math.radians(1.0) / (N * pri)
    points = np.array([(0.0, 0.0), (8.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (0.0, -8.0)],
                      dtype=float)

    radar = Radar(c=SPEED_OF_LIGHT, f_c=fc, Xmax=Xmax, Ymax=Xmax,
                  spectr_w=B, ph_c=0.0, ang=9.0, nifft_size=256)
    _, _, f_r, *_ = radar.base_img_params()
    R0 = 500_000.0

    # маржины guard'а ≥ 1.3 по C1a и C2 (плановое требование)
    checks = {c.code: c for c in check_applicability(radar, _SyntheticTarget(
        omega=omega, pri=pri, num_pulses=N, R0=R0, x0=points[:, 0],
        y0=points[:, 1]))}
    assert checks["C1a"].margin >= 1.3
    assert checks["C2"].margin >= 1.3

    E = _E_from_points(f_r, omega, pri, N, points, R0)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, _ = azimuth_compress(P.astype(np.complex128), omega, fc, pri=pri)
    amp = compute_amplitude(I)

    psf_x = 0.886 * SPEED_OF_LIGHT / (2 * B)                    # dr
    psf_y = 0.886 * (SPEED_OF_LIGHT / fc) / (2 * omega * N * pri)
    dx_tol, dy_tol = 2 * psf_x, 2 * psf_y

    peaks = _local_peak_positions(amp, range_axis, azimuth_axis)
    for m_t, y_t in points:
        m_p, y_p = _nearest_peak(m_t, y_t, peaks)
        assert abs(m_p - m_t) <= dx_tol, (m_t, y_t, m_p, y_p, dx_tol)
        assert abs(y_p - y_t) <= dy_tol, (m_t, y_t, m_p, y_p, dy_tol)


# ------------------------------------- 2) стандартный тракт, solarB_high ----

def test_standard_truth_solarB_top5(radar, target):
    """Реальная модель из фикстур: топ-5 по интенсивности — позиции верны."""
    _, _, f_r, *_ = radar.base_img_params()
    R0 = target.R0

    E, _ = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, _ = azimuth_compress(
        P.astype(np.complex128), target.omega, radar.f_c, pri=target.pri)
    amp = compute_amplitude(I)

    sc = target._load_scatterers()
    top = np.argsort(sc["intens"])[::-1][:5]

    psf_x = 0.886 * SPEED_OF_LIGHT / (2 * radar.spectr_w)
    psf_y = 0.886 * (SPEED_OF_LIGHT / radar.f_c) / (2 * target.omega * target.num_pulses * target.pri)
    dx_tol, dy_tol = 2 * psf_x, 2 * psf_y

    peaks = _local_peak_positions(amp, range_axis, azimuth_axis)
    for i in top:
        m_p, y_p = _nearest_peak(sc["x0"][i], sc["y0"][i], peaks)
        assert abs(m_p - sc["x0"][i]) <= dx_tol, (i, m_p, sc["x0"][i])
        assert abs(y_p - sc["y0"][i]) <= dy_tol, (i, y_p, sc["y0"][i])


# ------------------------------------------------- 3) полярный тракт, 5 точек

def test_polar_truth_5points():
    """Полярный тракт, Δθ=15°: средняя |err_x| ≤ 0.3·dx (F9: 2.8 см при dx≈10 см)."""
    fc, B = 9.5e9, 1.5e9
    pri, N = 5e-4, 1000                       # T_obs = 0.5 c → Δθ = 15°
    omega = math.radians(15.0) / (N * pri)
    Xmax = Ymax = 40.0
    R0 = 500_000.0
    points = np.array([(0.0, 0.0), (8.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (0.0, -8.0)],
                      dtype=float)

    radar = Radar(c=SPEED_OF_LIGHT, f_c=fc, Xmax=Xmax, Ymax=Ymax,
                  spectr_w=B, ph_c=0.0, ang=9.0, nifft_size=256)
    _, _, f_r, *_ = radar.base_img_params()

    E = _E_from_points(f_r, omega, pri, N, points, R0)

    kspace = build_kspace_grid(f_r, omega, pri, N)
    k, theta = kspace["k"], kspace["theta"]
    grid = design_cartesian_grid(k, theta, Xmax, Ymax)
    E_cart = interpolate_polar_to_cartesian(E, k, theta, grid, method="linear")
    blind = process_blind_zones(E_cart, grid["kx_axis"], grid["ky_axis"],
                                window_type="taylor", apply_zero_pad=True,
                                pad_factor=2, nbar=4, sll=-35.0)
    I, x_axis, y_axis, x_span, y_span = form_image_2d(
        blind["E_final"], grid["dkx"], grid["dky"])
    amp = np.abs(I)

    dr = SPEED_OF_LIGHT / (2 * B)     # дальностное разрешение (10 см при B=1.5 ГГц)
    peaks = _local_peak_positions(amp, x_axis, y_axis)
    errs = []
    for m_t, y_t in points:
        m_p, y_p = _nearest_peak(m_t, y_t, peaks)
        assert abs(y_p - y_t) <= 2.0 * (2 * 0.886 * (SPEED_OF_LIGHT / fc) / (2 * math.radians(15.0))), \
            (m_t, y_t, m_p, y_p)
        errs.append(abs(m_p - m_t))
    assert np.mean(errs) <= 0.3 * dr, (errs, dr)


# -------------------------------------------- 4) canary: алиасинг = ложь ----

class _OldAliasConfig:
    """Старый дефолт демо-1: N=16, PRI=1 c, Δθ=9°, survey 20 — C1a/C2 fail."""
    omega = math.radians(9.0) / 16.0
    pri = 1.0
    num_pulses = 16
    R0 = 500_000.0
    V = 0.0
    alpha = 0.0


def test_canary_aliasing_corrupts_positions():
    """Гарантия «зубастости» тестов 1–3: в алиас-режиме позиции ЛОЖНЫ."""
    from simulation.applicability import hard_failures

    fc, B, X = 8e9, 0.5e9, 20.0
    omega, pri, N, R0 = _OldAliasConfig.omega, _OldAliasConfig.pri, \
        _OldAliasConfig.num_pulses, _OldAliasConfig.R0

    radar = Radar(c=SPEED_OF_LIGHT, f_c=fc, Xmax=X, Ymax=X,
                  spectr_w=B, ph_c=0.0, ang=9.0, nifft_size=256)
    _, _, f_r, *_ = radar.base_img_params()
    target = Target(MATRIX, pri=pri, num_pulses=N, R0=R0, V=0.0, alpha=0.0,
                    omega=omega)

    # Guard обязан репортить C1a (совместно с test_applicability)
    checks = check_applicability(radar, target)
    c1a = [c for c in checks if c.code == "C1a"][0]
    assert not c1a.ok and c1a.severity == "hard"

    E, _ = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, _ = azimuth_compress(P.astype(np.complex128), omega, fc, pri=pri)
    amp = compute_amplitude(I)

    sc = target._load_scatterers()
    top = np.argsort(sc["intens"])[::-1][:5]

    psf_y = 0.886 * (SPEED_OF_LIGHT / fc) / (2 * omega * N * pri)   # ≈1.06 м
    peaks = _local_peak_positions(amp, range_axis, azimuth_axis)
    errs_y = []
    for i in top:
        _, y_p = _nearest_peak(sc["x0"][i], sc["y0"][i], peaks)
        errs_y.append(abs(y_p - sc["y0"][i]))
    # ХУДШИЙ из пяти уходит за 3·PSF — часть рассеивателей легла на
    # aliased-позиции; даже ближайший к каждому — ложный (алиас ×9–12)
    assert max(errs_y) > 3.0 * psf_y, (errs_y, 3 * psf_y)
