"""Тесты воспроизводимости: seed управляет всеми случайными эффектами."""
import numpy as np

from simulation.raw_generator import SimEffects, generate_raw_matrix


def test_same_seed_same_matrix(radar, target):
    eff = SimEffects(use_complex_sigma=True, use_anisotropy=True,
                     use_shadowing=True, use_antenna_pattern=True,
                     use_phase_noise=True, seed=42)
    E1, _ = generate_raw_matrix(radar, target, eff)
    E2, _ = generate_raw_matrix(radar, target, eff)
    assert np.array_equal(E1, E2)


def test_different_seed_different_matrix(radar, target):
    E1, _ = generate_raw_matrix(radar, target, SimEffects(use_complex_sigma=True, seed=1))
    E2, _ = generate_raw_matrix(radar, target, SimEffects(use_complex_sigma=True, seed=2))
    assert not np.array_equal(E1, E2)


def test_independent_sub_rngs(radar, target):
    """Включение фазового шума не меняет вклад случайных фаз рассеивателей."""
    E_base, _ = generate_raw_matrix(radar, target, SimEffects(use_complex_sigma=True, seed=7))
    E_pn, _ = generate_raw_matrix(
        radar, target, SimEffects(use_complex_sigma=True, use_phase_noise=True, seed=7))
    E_pn0, _ = generate_raw_matrix(
        radar, target, SimEffects(use_complex_sigma=True, use_phase_noise=True,
                                  phase_noise_std_deg=1e-9, seed=7))
    # С пренебрежимо малым шумом сигнал совпадает с базовым
    assert np.allclose(E_pn0, E_base, atol=1e-9)
    # С заметным шумом — отличается, но подгенератор фаз тот же
    assert not np.allclose(E_pn, E_base)
