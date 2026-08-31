"""Регрессионные тесты: выключенные эффекты == классическая модель."""
import numpy as np

from simulation.raw_generator import SimEffects, generate_raw_matrix


def test_effects_off_matches_reference(radar, target, reference_matrix):
    """E при effects=None и при SimEffects() побитово равна старой формуле."""
    E_none, _ = generate_raw_matrix(radar, target)
    E_off, _ = generate_raw_matrix(radar, target, SimEffects())
    assert np.array_equal(E_none, E_off)
    assert np.array_equal(E_none, reference_matrix)


def test_kinematics_linear_when_nonlinear_params_zero(target):
    """Ускорение/вибрация = 0 не меняют траекторию."""
    from models.target import Target
    sc = target._load_scatterers()
    t_lin = Target(target.file, pri=1.0, num_pulses=4, R0=100.0,
                   V=10.0, alpha=30.0, omega=0.1)
    t_nl = Target(target.file, pri=1.0, num_pulses=4, R0=100.0,
                  V=10.0, alpha=30.0, omega=0.1, accel=0.0, vib_amplitude=0.0)
    for n in range(4):
        xa, ya = t_lin._absolute_positions(t_lin.slow_time(n), sc["x0"], sc["y0"])
        xb, yb = t_nl._absolute_positions(t_nl.slow_time(n), sc["x0"], sc["y0"])
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)


def test_demo_matrices_unchanged(radar, target, reference_matrix):
    """Полный прогон демо-сценария даёт сфокусированное РЛИ как раньше."""
    from processing.range_compress import range_compress
    from processing.azimuth_compress import azimuth_compress

    E, f_r = generate_raw_matrix(radar, target)
    P, _, range_axis, _ = range_compress(E, f_r, R0=target.R0)
    I, _, _ = azimuth_compress(P.astype(np.complex128), target.omega, radar.f_c)
    E_ref, _ = generate_raw_matrix(radar, target, SimEffects())
    P_ref, _, _, _ = range_compress(E_ref, f_r, R0=target.R0)
    I_ref, _, _ = azimuth_compress(P_ref.astype(np.complex128), target.omega, radar.f_c)
    assert np.array_equal(I, I_ref)
