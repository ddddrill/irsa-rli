"""Интеграционный тест: полный конвейер со всеми эффектами реализма."""
import numpy as np
import pytest

from simulation.raw_generator import SimEffects, generate_raw_matrix


ALL_ON = dict(
    use_complex_sigma=True, use_anisotropy=True, use_shadowing=True,
    use_antenna_pattern=True, use_phase_noise=True, seed=42,
)


def _run_standard(E, f_r, target, radar, pri):
    from processing.range_compress import range_compress
    from processing.mocomp import mocomp
    from processing.azimuth_compress import azimuth_compress
    from processing.polar_reformat import compute_image_entropy

    P, _, range_axis, _ = range_compress(E, f_r, R0=target.R0)
    P_comp, shifts, _, phase_errors = mocomp(P, range_axis)
    I, _, _ = azimuth_compress(P_comp.astype(np.complex128), target.omega,
                               radar.f_c, pri=pri)
    assert np.isfinite(compute_image_entropy(I))
    assert np.isfinite(np.abs(I).sum()) and np.abs(I).sum() > 0
    return I


def test_standard_pipeline_with_all_effects(radar, target):
    E, f_r = generate_raw_matrix(radar, target, SimEffects(**ALL_ON))
    I = _run_standard(E, f_r, target, radar, pri=1.0)
    assert I.ndim == 2


def test_polar_pipeline_with_all_effects(radar):
    """Полярный конвейер (через DataProcessor) со всеми эффектами."""
    import math
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    omega = math.radians(9.0) / 64.0
    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="с полярным переформатированием",
        f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5, ang=9.0,
        range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=omega, pri=0.01, num_pulses=64,
        effects=SimEffects(**ALL_ON),
    )
    E, f_r, target = proc.generate_raw_data()
    I, amplitude, x_axis, y_axis, _ = proc._compute_isar_polar_sync(E, f_r, target)
    assert I.shape == amplitude.shape
    assert np.isfinite(amplitude).all()
    assert len(x_axis) == I.shape[0] and len(y_axis) == I.shape[1]


def test_nonlinear_motion_reaches_pipeline(radar):
    """Ускорение + вибрация реально меняют сигнал и обрабатываются конвейером."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    common = dict(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=9.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=0.005, pri=1.0, num_pulses=64,
    )
    p_lin = DataProcessor(**common)
    E_lin, f_r, t = p_lin.generate_raw_data()

    p_nl = DataProcessor(**common, accel=2.0, vib_amplitude=0.005, vib_frequency=2.0)
    E_nl, _, _ = p_nl.generate_raw_data()
    assert not np.allclose(E_lin, E_nl)

    I_lin = _run_standard(E_lin, f_r, t, radar, pri=1.0)
    I_nl = _run_standard(E_nl, f_r, p_nl._make_target(), radar, pri=1.0)
    assert I_lin.shape == I_nl.shape


def test_data_processor_backward_compatible_kwargs(radar):
    """DataProcessor без новых kwargs работает как раньше."""
    pytest.importorskip("PyQt5")
    from processing.data_processor import DataProcessor

    proc = DataProcessor(
        filename="matrices/matrices_solarB_high.pkl",
        method="стандартный", f_c=8.0, Xmax=20.0, Ymax=20.0, spectr_w=0.5,
        ang=9.0, range_m=500000.0, nifft_size=256,
        V=0.0, alpha=0.0, omega=0.005, pri=1.0, num_pulses=16,
    )
    E, f_r, _ = proc.generate_raw_data()
    assert E.shape[1] == 16 and np.isfinite(E).all()
