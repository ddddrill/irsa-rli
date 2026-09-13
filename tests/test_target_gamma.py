"""Тесты задачи R8a: угловое ускорение γ в модели цели (C&M, (1.13)).

θ(t) = θ₀ + Ω·t + γ·t²/2
Дефолт γ = 0 → побитовое совпадение с текущей реализацией.
"""
import math

import numpy as np
import pytest

from models.target import Target


def test_gamma_zero_bitwise_compat(target, radar):
    """gamma=0 → позиции, дальности, углы и E-матрица побитово совпадают."""
    from simulation.raw_generator import generate_raw_matrix

    sc = target._load_scatterers()
    t = Target(
        target.file, pri=target.pri, num_pulses=target.num_pulses,
        R0=target.R0, V=target.V, alpha=target.alpha,
        omega=target.omega, gamma=0.0,
    )
    for n in range(target.num_pulses):
        t_n = target.slow_time(n)
        xa, ya = target._absolute_positions(t_n, sc["x0"], sc["y0"])
        xb, yb = t._absolute_positions(t_n, sc["x0"], sc["y0"])
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    assert np.array_equal(target.get_ranges_all(), t.get_ranges_all())
    assert np.array_equal(target.get_aspect_angles(), t.get_aspect_angles())

    E_ref, _ = generate_raw_matrix(radar, target)
    E_new, _ = generate_raw_matrix(radar, t)
    assert np.array_equal(E_ref, E_new)


def test_theta_formula():
    """Случайные (omega, gamma, pri): углы == omega*t + 0.5*gamma*t²."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        omega = rng.uniform(0.001, 0.2)
        gamma = rng.uniform(-1e-4, 1e-4)
        pri = rng.uniform(0.01, 0.1)
        num_pulses = int(rng.integers(4, 64))

        t = Target(
            "matrices/matrices_solarB_high.pkl", pri=pri,
            num_pulses=num_pulses, R0=0.0, V=0.0, alpha=0.0,
            omega=omega, gamma=gamma,
        )
        n = np.arange(num_pulses)
        expected = omega * n * pri + 0.5 * gamma * (n * pri) ** 2
        assert np.allclose(t.get_aspect_angles(), expected, rtol=1e-12, atol=0.0)

        # Проверить _absolute_positions через slow_time: одиночный рассеиватель
        # на оси X (x0=1, y0=0, X_cm=Y_cm=0) повернут ровно на θ(t_n):
        #   X_abs = cos(θ), Y_abs = sin(θ)  →  θ = atan2(Y_abs, X_abs).
        x0, y0 = np.array([1.0]), np.array([0.0])
        for k in range(num_pulses):
            t_n = t.slow_time(k)
            X, Y = t._absolute_positions(t_n, x0, y0)
            theta_impl = math.atan2(float(Y[0]), float(X[0]))
            theta_expect = omega * t_n + 0.5 * gamma * t_n ** 2
            assert math.isclose(theta_impl, theta_expect, rel_tol=1e-9, abs_tol=1e-9)


def test_polar_pipeline_survives_gamma(radar):
    """gamma=1e-4: полярный конвейер выживает (расфокусировка документирована)."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor
    from processing.polar_reformat import compute_image_entropy

    omega = math.radians(9.0) / 64.0
    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="с полярным переформатированием",
        f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5, ang=9.0,
        range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=omega, pri=0.01, num_pulses=64,
        gamma=1e-4,
    )
    E, f_r, target = proc.generate_raw_data()
    I, amplitude, x_axis, y_axis, _ = proc._compute_isar_polar_sync(E, f_r, target)

    assert np.isfinite(compute_image_entropy(I))
    assert np.isfinite(amplitude).all()
    assert np.isfinite(I).all()
    # Оси в метрах (размер сцены ~ десятки метров), не в отсчётах.
    assert 1.0 < (x_axis[-1] - x_axis[0]) < 1e5
    assert 1.0 < (y_axis[-1] - y_axis[0]) < 1e5


def test_build_kspace_grid_theta_array(radar):
    """build_kspace_grid с theta_array следует нелинейной кинематике.

    Опциональный параметр (R8a, п.8) — нужен R6 для полярки при γ≠0.
    По умолчанию сетка остаётся линейной (обратная совместимость).
    """
    from processing.polar_reformat import build_kspace_grid

    pri = 0.05
    num_pulses = 32
    omega, gamma = 0.01, 2e-4
    t = Target(
        "matrices/matrices_solarB_high.pkl", pri=pri,
        num_pulses=num_pulses, omega=omega, gamma=gamma,
    )
    _, _, f_r, *_ = radar.base_img_params()

    kspace = build_kspace_grid(
        f_r, omega, pri, num_pulses, theta_array=t.get_aspect_angles()
    )
    assert np.allclose(kspace["theta"], t.get_aspect_angles())

    kspace_lin = build_kspace_grid(f_r, omega, pri, num_pulses)
    assert np.allclose(
        kspace_lin["theta"], omega * np.arange(num_pulses) * pri
    )
    assert not np.allclose(kspace_lin["theta"], kspace["theta"])