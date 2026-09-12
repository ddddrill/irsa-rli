"""Тесты выравнивания огибающей v2 (`processing.mocomp.envelope_alignment`).

Задача R2 плана исправлений (§5): устранение трёх дефектов as-is —
целочисленные лаги, опора = первый профиль, np.roll. Проверяются:
  * суббиновая точность (RMSE < 0.05 бина на линейном walk);
  * отсутствие заворота энергии при pad-and-shift;
  * устойчивость усреднённого опорного профиля vs первого;
  * legacy-режим (ref_mode="first", subbin=False) даёт те же лаги, что
    старая np.roll-реализация.

Допущения сценариев:
  * тесты 1–3 моделируют одиночный рассеиватель напрямую в E-домене:
    E[:, n] = exp(-j·4π·f/c·R(t)), R(t) = R0 + V·t; сжатие по дальности —
    `range_compress` с окном kaiser(β=8) (C&M, §6.3.1). Параболическая
    интерполяция пика корреляции точна для оконных огибающих (~RMSE 0.03);
    для boxcar-ПФР смещение параболы достигает ~0.2 бина (план §5.2.3).
  * шум и «вспышка» (нетипичный первый импульс) детерминированы seed.
"""
import math

import numpy as np
from scipy.signal import correlate

from processing.mocomp import envelope_alignment
from processing.range_compress import range_compress

SPEED_OF_LIGHT = 3e8


def _point_scenario(num_pulses=64, V=0.3, pri=1.0, M=256,
                    window="kaiser", window_param=8.0,
                    r0_off_bins=0.0, snr_db=None,
                    notch_first=False, seed=0):
    """Одиночный точечный рассеиватель с линейным range walk.

    Возвращает (P, dr): матрицу профилей дальности M×N и разрешение по
    дальности. Физический сдвиг бина n равен V·n·pri/dr (дробно).
    """
    f_c, B, R0 = 8e9, 0.5e9, 500000.0
    f = np.linspace(f_c - B / 2, f_c + B / 2, M)
    t = np.arange(num_pulses) * pri
    dr = SPEED_OF_LIGHT / (2.0 * M * ((f[-1] - f[0]) / (M - 1)))

    R = R0 + r0_off_bins * dr + V * t
    E = np.exp(-1j * 4.0 * np.pi * f[:, None] / SPEED_OF_LIGHT * R[None, :])

    if snr_db is not None:
        rng = np.random.default_rng(seed)
        ps = np.mean(np.abs(E) ** 2)
        pn = ps / (10.0 ** (snr_db / 10.0))
        E = E + (rng.normal(0, 1, E.shape) + 1j * rng.normal(0, 1, E.shape)) * math.sqrt(pn / 2)
    if notch_first:
        # нетипичный первый импульс: половина спектра вырезана -> широкая
        # «расплывчатая» огибающая, плохо пригодная как опора (план §5.2.2)
        half = M // 2
        E[:half, 0] = 0.0

    P, _, _, dr = range_compress(E, f, R0=R0, window=window, window_param=window_param)
    return P, dr


def _integer_walk_profile(M=256, ks=(-1, 1, 2, -3, 3, 4, 0, 5)):
    """Колонки — точные целочисленные сдвиги первого профиля (для legacy-теста).

    Возвращает (P, ks): матрицу M×N, где колонка n = roll(P[:, 0], ks[n]).
    """
    f_c, B, R0 = 8e9, 0.5e9, 500000.0
    f = np.linspace(f_c - B / 2, f_c + B / 2, M)
    E = np.exp(-1j * 4.0 * np.pi * f[:, None] / SPEED_OF_LIGHT * R0)
    P0, _, _, _ = range_compress(E, f, R0=R0)
    ref = P0[:, 0]
    N = len(ks)
    P = np.empty((M, N), dtype=np.complex128)
    for n, k in enumerate(ks):
        P[:, n] = np.roll(ref, k)
    return P, np.asarray(ks, dtype=float)


def _legacy_lags(P):
    """Лаги старой np.roll-реализации: argmax(complex-correlate) - (M-1)."""
    M, N = P.shape
    lags = np.zeros(N, dtype=float)
    for n in range(1, N):
        corr = correlate(P[:, n], P[:, 0], mode="full")
        lags[n] = np.argmax(corr) - (M - 1)
    return lags


def test_subbin_alignment_precision():
    """Тест §5.5.1: суббиновое выравнивание на линейном walk (V=0.3 м/с, N=64).

    Истинный сдвиг профиля n равен V·t/dr (дробно). Допуск RMSE < 0.05 бина.
    """
    num_pulses, V, pri = 64, 0.3, 1.0
    P, dr = _point_scenario(num_pulses=num_pulses, V=V, pri=pri)
    P_aligned, shifts = envelope_alignment(P)

    assert P_aligned.shape == P.shape
    assert shifts.dtype.kind == "f" and shifts.ndim == 1 and shifts.shape[0] == num_pulses

    expected = V * np.arange(num_pulses) * pri / dr
    rmse = math.sqrt(float(np.mean((shifts - expected) ** 2)))
    assert rmse < 0.05, f"RMSE(subbin) = {rmse:.4f} >= 0.05 бина"


def test_no_wrap_energy_conservation():
    """Тест §5.5.2: рассеиватель у края сетки (R-R0 ≈ 0.8·M·dr/2).

    pad-and-shift не заворачивает энергию (нет фантома у противоположного
    края) и сохраняет энергию каждого столбца (Парсеваль).
    """
    M, num_pulses, V = 256, 16, 0.08
    P, dr = _point_scenario(M=M, num_pulses=num_pulses, V=V, r0_off_bins=0.8 * M / 2)

    P_aligned, shifts = envelope_alignment(P)
    ref_bin = int(np.argmax(np.abs(P[:, 0])))
    aspect = np.argmax(np.abs(P_aligned), axis=0)
    assert np.all(aspect == ref_bin), "пики не встали на опорную линию"

    P_big, _ = envelope_alignment(P, guard_bins=64)
    rel_diff = np.abs(P_aligned - P_big).max() / np.abs(P).max()
    assert rel_diff < 1e-4, \
        f"меньший guard меняет результат (rel={rel_diff:.2e}) — заворот энергии не поглощён"

    l1_in = np.abs(P).sum(axis=0)
    l1_out = np.abs(P_aligned).sum(axis=0)
    assert np.allclose(l1_out, l1_in, rtol=1e-3), "L1-энергия не сохранилась"

    l2_in = (np.abs(P) ** 2).sum(axis=0)
    l2_out = (np.abs(P_aligned) ** 2).sum(axis=0)
    assert np.allclose(l2_out, l2_in, rtol=1e-4), "L2-энергия не сохранилась"


def test_average_reference_more_robust_than_first():
    """Тест §5.5.3: усреднённая опора не хуже первого профиля при AWGN.

    Сценарий: одиночный рассеиватель + AWGN (0 дБ) + нетипичный первый
    импульс (вырезана половина спектра — широкая огибающая, §5.2.2(a)-(b)).
    Ожидается RMSE(ref_mode="average") <= RMSE(ref_mode="first").
    """
    num_pulses, V, pri = 64, 0.3, 1.0
    P, dr = _point_scenario(num_pulses=num_pulses, V=V, pri=pri,
                            snr_db=0.0, notch_first=True, seed=303)
    expected = V * np.arange(num_pulses) * pri / dr

    _, shifts_avg = envelope_alignment(P, ref_mode="average")
    _, shifts_first = envelope_alignment(P, ref_mode="first")

    rmse_avg = math.sqrt(float(np.mean((shifts_avg - expected) ** 2)))
    rmse_first = math.sqrt(float(np.mean((shifts_first - expected) ** 2)))
    assert rmse_avg <= rmse_first, \
        f"average ({rmse_avg:.4f}) хуже first ({rmse_first:.4f})"


def test_legacy_first_integer_lags_match_old_roll():
    """Тест §5.5.5: legacy-режим (subbin=False, ref_mode="first").

    Сдвиги совпадают с лагами старой np.roll-реализации на целочисленном
    walk (матрицы могут отличаться из-за pad-and-shift — сравниваем лаги).
    """
    P, ks = _integer_walk_profile()
    _, shifts = envelope_alignment(P, subbin=False, ref_mode="first")

    old = _legacy_lags(P)
    assert np.allclose(shifts, old, atol=1e-12), "лаги не совпали со старыми"