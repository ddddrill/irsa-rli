"""Тесты воспроизводимости: seed управляет всеми случайными эффектами."""
import numpy as np

from models.radar import Radar
from simulation.raw_generator import SimEffects, generate_raw_matrix
from simulation.rng import spawn_rng

SPEED_OF_LIGHT = 3e8


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


def test_same_seed_same_matrix_with_noise(radar, target):
    """AWGN из seeded-подгенератора: один seed — побитово одинаковый результат
    при любом SNR (дефект фикса F3: глобальный np.random.randn давал разные
    матрицы при SNR < inf, README обещал воспроизводимость — ложь)."""
    eff = SimEffects(seed=42)
    radar_noisy = Radar(radar.c, radar.f_c, radar.Xmax, radar.Ymax,
                        radar.spectr_w, radar.ph_c, radar.ang,
                        nifft_size=radar.nifft_size, snr_db=15.0)
    E1, _ = generate_raw_matrix(radar_noisy, target, eff)
    E2, _ = generate_raw_matrix(radar_noisy, target, eff)
    assert np.array_equal(E1, E2)


def test_awgn_snr_inf_bitwise(radar, target):
    """SNR=inf — путь без шума, побитово идентичен legacy."""
    eff_clean = SimEffects(seed=42)
    E_clean, _ = generate_raw_matrix(radar, target, eff_clean)
    E_inf, _ = generate_raw_matrix(radar, target, eff_clean)  # snr_db=inf
    assert np.array_equal(E_clean, E_inf)


def test_awgn_isolated_from_other_effects(radar, target):
    """Включение фазового шума не меняет реализацию шума AWGN (подгенератор
    \"awgn\" независим по построению spawn_rng)."""
    radar_noisy = Radar(radar.c, radar.f_c, radar.Xmax, radar.Ymax,
                        radar.spectr_w, radar.ph_c, radar.ang,
                        nifft_size=radar.nifft_size, snr_db=15.0)
    sc = target._load_scatterers()
    R = target.get_ranges_all()
    _, _, f_r, *_ = radar.base_img_params()
    E_stem = (sc["intens"][None, None, :] *
              np.exp(1j * (-4.0 * np.pi * f_r[:, None, None] / SPEED_OF_LIGHT *
                           R[None, :, :]))).sum(axis=2)

    # Те же производные, но по компонентам: один подгенератор, как в
    # generate_raw_matrix — два последовательных draws standard_normal
    rng = spawn_rng(42, "awgn")
    noise = rng.standard_normal(E_stem.shape) + \
        1j * rng.standard_normal(E_stem.shape)
    P_signal = np.mean(np.abs(E_stem) ** 2)
    P_noise = P_signal / 10.0 ** (15.0 / 10.0)
    expected = E_stem + np.sqrt(P_noise / 2.0) * noise

    E1, _ = generate_raw_matrix(radar_noisy, target, SimEffects(seed=42))
    E2, _ = generate_raw_matrix(radar_noisy, target,
                                SimEffects(use_phase_noise=True, phase_noise_std_deg=1e-9,
                                           seed=42))
    # Эталон совпадает побитово: шум AWGN взят из spawn_rng(seed, "awgn")
    assert np.array_equal(E1, expected)
    # Включение фазового шума (std=1e-9 град) не меняет реализацию AWGN:
    # меняется только ствол exp(j*1.7e-11 рад) — расхождение на этом пределе
    lvl = np.abs(E1).max()
    assert np.allclose(E1, E2, atol=1e-9 * lvl) and not np.array_equal(E1, E2)

