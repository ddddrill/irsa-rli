"""Keystone-перешкалирование медленного времени (задача R6, §9 плана).

Keystone убирает line-of-sight миграцию дальности вращающегося рассеивателя
(член −y·sin(ω·t) ≈ −y·ω·t в R(t)) до азимутального сжатия — C&M, §5.5,
(5.30)/(5.37): перешкалирование медленного времени t_ks = (f_c/f_r)·t
делает фазу наклонной дальности независимой от частоты f_r.

Внимание к направлению (§9.2.4): ВЫХОДНОЙ отсчёт j (t_ks = j·PRI) строки m
берётся из ИСХОДНОЙ строки по дробному индексу j·(f_c/f_r[m]) — НЕ
j·(f_r[m]/f_c). Обратная операция (индексы j·(f_r/f_c)) не фокусирует:
тест 9.6.1 фиксирует знак страховкой ошибки ~1.

Keystone компенсирует только вращательный линейный член миграции
(rotational linear migration). Поступательное движение (V ≠ 0) выравнивает
MOCOMP; квадратичный остаток (член γ·t², дальностный чирп) остаётся для
дальностных миграционных алгоритмов типа RID (§9.2.8).

Ресемплинг строки (все операции векторизованы, циклов по N нет):
  * method="sinc" (основной) — band-limited: FFT-оверсемплинг ×16 строки
    с нулевой приставкой + локальная 4-точечная кубическая интерполяция на
    оверсемпленной сетке. Обладает той же точностью, что точное
    (Dirichlet/периодический sinc) преобразование, при стоимости ~1/15.
    Ошибка на «неутекающих» тонах < 1e-6.
  * method="cubic" (fallback) — scipy.interpolate.CubicSpline на исходной
    строке (комплексный ряд интерполируется как есть, подряд по строкам).
"""
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline

__all__ = ["keystone_transform"]

# Кратность FFT-оверсемплинга строки (перекрывает частоты строк до 0.5·fs).
_OVERSAMPLE = 16


def keystone_transform(E, f_r, f_c, method="sinc", pad=None):
    """Keystone-перешкалирование медленного времени. C&M, §5.5 (5.30)/(5.37).

    Выходной отсчёт j (t_ks = j·PRI) строки m берётся из исходной строки
    по ДРОБНОМУ индексу j·(f_c/f_r[m])  [НЕ j·(f_m/f_c)! §9.2.4].
    f_r — абсолютные частоты (Гц). pad: нулевая приставка; авто —
    ceil(N·(f_c/min(f_r) − 1)) + 2 (максимум выхода за сетку ~N·B/(2f_c)).
    Returns: E_ks (M×N, та же сетка).

    Args:
        E: сырые данные M×N (частоты × импульсы).
        f_r: вектор абсолютных частот (M,), Гц.
        f_c: центральная частота радара, Гц.
        method: "sinc" (band-limited, основной) или "cubic" (fallback).
        pad: размер нулевой приставки слева/справа; None → авто
            ceil(N·(f_c/min(f_r) − 1)) + 2.

    Returns:
        E_ks: перешкалированная матрица (M×N) — каждая строка ресемплирована
        на равномерной если медленного времени j·PRI.

    Raises:
        ValueError: если method не "sinc"/"cubic".
    """
    if method not in ("sinc", "cubic"):
        raise ValueError(f"method='{method}' не поддерживается (sinc|cubic)")

    E = np.asarray(E, dtype=np.complex128)
    f_r = np.asarray(f_r, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"E должна быть M×N, получена форма {E.shape}")
    if f_r.ndim != 1 or f_r.shape[0] != E.shape[0]:
        raise ValueError(
            f"f_r должна быть вектором длины M={E.shape[0]}, форма {f_r.shape}"
        )
    if f_r.min() <= 0.0:
        raise ValueError("f_r должны быть положительными частотами")

    M, N = E.shape
    ratio = f_c / f_r                      # (M,)
    if pad is None:
        pad = int(np.ceil(N * (np.max(ratio) - 1.0))) + 2
    pad = max(int(pad), 0)
    N2 = N + 2 * pad

    # Ряд с нулевой приставкой (слева и справа) для симметрии (§9.4.1).
    E_pad = np.zeros((M, N2), dtype=np.complex128)
    E_pad[:, pad:pad + N] = E

    # Индексы ресемплинга: выходной отсчёт j = входной j·(f_c/f_r[m]) + pad.
    idx = ratio[:, None] * np.arange(N)[None, :] + pad

    if method == "cubic":
        return _resample_cubic(E_pad, idx)

    return _resample_sinc(E_pad, idx)


def _resample_sinc(E_pad, idx):
    """Band-limited ресемплинг строк: FFT-оверсемплинг ×16 + локальный кубик.

    Строка (с приставкой) переводится в спектр, спектр расширяется нулями
    в ×16 длину (переставлены половинки — отрицательные частоты в конец),
    обратным БПФ получается exact 16-кратно оверсемпленный ряд (band-limited
    интерполяция той же полосы). Дробная часть idx·16 добирается локальной
    4-точечной кубической интерполяцией на оверсемпленной сетке — на тоне с
    несущей f0 ошибка ~ (2π·f0/16)^4-го порядка.

    Все операции — матричные (по осям строк и отсчётов), циклов по N нет.
    """
    M, N2 = E_pad.shape
    L = _OVERSAMPLE

    spec = fft(E_pad, axis=1)
    half = N2 // 2
    os_spec = np.zeros((M, L * N2), dtype=np.complex128)
    os_spec[:, :half] = spec[:, :half]
    os_spec[:, -(N2 - half):] = spec[:, half:]
    row_os = ifft(os_spec, axis=1) * L

    X = idx * L
    Xb = np.floor(X).astype(np.int64)
    u = X - Xb

    out = np.zeros(idx.shape, dtype=np.complex128)
    rows = np.arange(M)[:, None]
    for tap in range(-1, 3):
        # Вес Лагранжа 4-точечного кубика для узла tap при смещении u∈[0,1)
        w = np.ones_like(u)
        for node in range(-1, 3):
            if node != tap:
                w *= (u - node) / (tap - node)
        kc = np.clip(Xb + tap, 0, L * N2 - 1)
        out += row_os[rows, kc] * w
    return out


def _resample_cubic(E_pad, idx):
    """Fallback: по-строчный CubicSpline на исходной строке с приставкой.

    Комплексный ряд интерполируется как есть (scipy поддерживает complex>):
    CubicSpline(np.arange(N+2p), row_padded)(idx). Точность на низких
    «демо-частотах» ≤ 1e-2 (§9.6.1); растёт с частотой как (ω·h)^4.
    """
    M = E_pad.shape[0]
    N2 = E_pad.shape[1]
    out = np.empty(idx.shape, dtype=np.complex128)
    x = np.arange(N2)
    for m in range(M):
        out[m] = CubicSpline(x, E_pad[m])(idx[m])
    return out