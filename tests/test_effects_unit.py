"""Юнит-тесты генераторов эффектов реализма."""
import math

import numpy as np

from simulation.scatterer_effects import (
    anisotropy_weights,
    make_complex_amplitudes,
    occlusion_mask,
)
from simulation.system_effects import phase_noise_process, two_way_antenna_weights


def test_complex_amplitudes_preserve_magnitude():
    intens = np.array([6.0, 25.0, 167.0, 1.0])
    sigma = make_complex_amplitudes(intens, seed=42)
    assert np.allclose(np.abs(sigma), intens)
    # Фазы действительно случайные, но воспроизводимые
    sigma2 = make_complex_amplitudes(intens, seed=42)
    assert np.array_equal(sigma, sigma2)


def test_anisotropy_weights_peak_and_shape():
    theta = np.linspace(0.0, 0.5, 21)  # содержит 0.25 точно
    theta0 = np.full(5, 0.25)  # все пики в центре сектора
    W = anisotropy_weights(theta0, theta, theta_c=0.05)
    assert W.shape == (21, 5)
    assert np.all(W > 0) and np.all(W <= 1.0)
    # На пиковом ракурсе вес ~1
    i_center = np.argmin(np.abs(theta - 0.25))
    assert np.allclose(W[i_center], 1.0, atol=1e-6)
    # Далеко от пика вес мал
    assert W[0].max() < 0.01


def test_occlusion_mask_shadows_behind_panel():
    # Точка 0 ближе всех и в той же "трубе" с точками 1 и 2
    ranges = np.array([[10.0, 12.0, 15.0]])
    y = np.array([[0.0, 0.1, 0.2]])
    vis = occlusion_mask(ranges, y, shadow_radius=0.4)
    assert vis[0, 0] and not vis[0, 1] and not vis[0, 2]
    # Точка в стороне — не затенена
    y2 = np.array([[0.0, 0.1, 5.0]])
    vis2 = occlusion_mask(ranges, y2, shadow_radius=0.4)
    assert vis2[0, 2]


def test_phase_noise_rms_matches_setting():
    phi = phase_noise_process(5000, 10.0, seed=42)
    assert abs(np.degrees(np.std(phi)) - 10.0) < 1e-6
    assert np.allclose(np.unwrap(phi), phi)  # непрерывный процесс
    # Детерминированность
    assert np.array_equal(phi, phase_noise_process(5000, 10.0, seed=42))


def test_antenna_weights_symmetric_and_bounded():
    w = two_way_antenna_weights(9.0, 101, pointing_error_deg=0.0, drift_deg=0.0)
    assert w.shape == (101,)
    assert np.all(w <= 1.0) and np.all(w > 0)
    assert np.allclose(w, w[::-1])  # симметрия при нулевой ошибке
    # Вес на краю ДН (-4.5 град) ~ 0.25 (двухпутевой -6 дБ)
    w_edge = two_way_antenna_weights(9.0, 2, pointing_error_deg=4.5)
    assert abs(w_edge[-1] - 0.25) < 0.02
    # Ошибка наведения гасит сигнал
    w_off = two_way_antenna_weights(9.0, 101, pointing_error_deg=9.0)
    assert w_off.max() < 0.01
