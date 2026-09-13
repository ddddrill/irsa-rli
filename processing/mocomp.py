"""Выравнивание огибающей профилей дальности (MOCOMP, шаг 1).

Задача R2 (план исправлений, §5): v2 выравнивания огибающей. Устранены
три дефекта as-is:
    1. опорный профиль по умолчанию — не первый, а усреднённый (двухпроход);
    2. целочисленные лаги заменены суббиновыми (параболическая интерполяция);
    3. np.roll (циркулярный сдвиг) заменён pad-and-shift (теорема о сдвиге
       в частотной области) — энергия не заворачивается через край сетки.
"""
import numpy as np
from scipy.signal import correlate


def _corr_lag(current, reference, subbin=True):
    """Лаг выравнивания по максимуму корреляции огибающих.

    Конвенция — как в v1: полный аргумент максимума корреляции
    `correlate(current, reference, mode="full")` минус (M-1); положительный
    лаг означает, что профиль `current` стоит правее опоры `reference`
    (в E-области «опережает» по дальности).

    Args:
        current: огибающая текущего профиля (M,).
        reference: огибающая опоры (M,).
        subbin: True — 3-точечная параболическая интерполяция пика
            ((C&M, (4.12)–(4.14); план §5.2.3, (5.2.2)); False — целый лаг.

    Returns:
        float: сдвиг в бинах.
    """
    corr = correlate(current, reference, mode="full").real
    i0 = int(np.argmax(corr))
    lag = float(i0 - (current.shape[0] - 1))
    if subbin and 0 < i0 < corr.shape[0] - 1:
        c_m, c_0, c_p = corr[i0 - 1], corr[i0], corr[i0 + 1]
        denom = c_m - 2.0 * c_0 + c_p
        if abs(denom) > 1e-12:
            lag += 0.5 * (c_m - c_p) / denom
    return lag


def _auto_guard(shifts):
    """Ширина защитной зоны: max|lags| + 8 (план §5.4, шаг 4)."""
    return int(np.ceil(float(np.max(np.abs(shifts))))) + 8


def _shift_columns(P, shifts, guard):
    """Некруглический сдвиг столбцов: FFT phase-ramp на сетке M2 = M + 2*guard.

    Профиль дальности — band-limited сигнал (полоса B радара), поэтому
    фазовый сдвиг в частотной области есть точная sinc-интерполяция
    (теорема о сдвиге, план §5.2.3, (5.2.3); C&M, (4.12)–(4.14)). Нулевые
    guard-поля поглощают заворот ДПФ: энергия, вышедшая за край столбца,
    не попадает в извлекаемое окно [guard : guard+M].

    Положительный сдвиг переносит содержимое столбца ВЛЕВО — конвенция
    лагов v1: лаг > 0 означает «профиль правее опоры, надо сдвинуть влево».

    Args:
        P: комплексная матрица профилей M×N.
        shifts: (N,) float — сдвиги в бинах на каждый столбец.
        guard: ширина защитной зоны (≥ max|shifts| + 2).

    Returns:
        P_aligned: комплексная матрица M×N (та же сетка дальности).
    """
    M, N = P.shape
    M2 = M + 2 * guard
    k = np.arange(M2)
    out = np.empty((M, N), dtype=np.complex128)
    for n in range(N):
        s = float(shifts[n])
        x = np.zeros(M2, dtype=np.complex128)
        x[guard:guard + M] = P[:, n]
        x = np.fft.ifft(np.fft.fft(x) * np.exp(+2j * np.pi * k * s / M2))
        out[:, n] = x[guard:guard + M]
    return out


def envelope_alignment(P, method="correlation", subbin=True,
                       ref_mode="average", guard_bins=None):
    """Выравнивание огибающих профилей дальности. C&M, §4.1.1 (v2).

    Корреляция по модулю |P| устойчива к фазовому шуму (план §5.2.2);
    суббиновый лаг — параболическая интерполяция пика корреляции + дробный
    сдвиг через FFT phase-ramp (pad-and-shift, план §5.2.3–§5.2.4). Дробный
    сдвиг вносит в E-домен предсказуемый линейный фазовый тренд
    4π·Δr_n·f/c (план §5.2.5) — PPP (фазовый автофокус) компенсирует его сам.

    Args:
        P: M×N комплексная матрица профилей (ось 0 — дальность).
        method: "correlation" (максимум корреляции огибающих) или
            "centroid" (разность центров тяжести огибающих).
        subbin: True — суббиновая точность (параболическая интерполяция
                максимума корреляции + дробный сдвиг через FFT phase-ramp).
        ref_mode: "average" — двухпроходная схема с усреднённым опорным
                  профилем (рекомендуется); "first" — legacy-режим (только
                  для A/B-сравнения в тестах).
        guard_bins: ширина нулевой «защитной» зоны при pad-and-shift
                    (по умолчанию auto: max|lags| + 8).

    Returns:
        P_aligned: M×N, та же сетка дальности.
        shifts: (N,) float — сдвиги в бинах (физически: shifts[n]·dr метров).
    """
    P = np.asarray(P, dtype=np.complex128)
    M, N = P.shape
    A = np.abs(P)

    if method == "correlation":
        shifts = np.empty(N, dtype=float)
        if ref_mode == "first":
            reference = A[:, 0].copy()
            for n in range(N):
                shifts[n] = _corr_lag(A[:, n], reference, subbin)
        elif ref_mode == "average":
            # проход 1: грубое выравнивание на первый профиль,
            # их среднее — «чистая» усреднённая опора (усиление √N)
            reference = A[:, 0].copy()
            coarse = np.empty(N, dtype=float)
            for n in range(N):
                coarse[n] = _corr_lag(A[:, n], reference, True)
            A1 = np.abs(_shift_columns(P, coarse, _auto_guard(coarse)))
            reference = A1.mean(axis=1)
            # проход 2: точное выравнивание исходных столбцов на опору
            for n in range(N):
                shifts[n] = _corr_lag(A[:, n], reference, subbin)
        else:
            raise ValueError(f"ref_mode={ref_mode!r}: ожидается 'average' или 'first'")
    elif method == "centroid":
        shifts = np.empty(N, dtype=float)
        ref_centroid = np.average(np.arange(M), weights=A[:, 0] ** 2 + 1e-30)
        for n in range(N):
            cur_centroid = np.average(np.arange(M), weights=A[:, n] ** 2 + 1e-30)
            shifts[n] = cur_centroid - ref_centroid
    else:
        raise ValueError(f"method={method!r}: ожидается 'correlation' или 'centroid'")

    if guard_bins is None:
        guard = _auto_guard(shifts)
    else:
        if guard_bins < np.max(np.abs(shifts)) + 2:
            raise ValueError(
                f"guard_bins={guard_bins} < max|shifts| + 2 ({np.max(np.abs(shifts)) + 2:.1f}) "
                "— заворот энергии дойдёт до окна извлечения"
            )
        guard = int(guard_bins)

    P_aligned = _shift_columns(P, shifts, guard)
    return P_aligned, shifts


def phase_autofocus(P_aligned, range_axis=None):
    """Коррекция фазы методом опорной точки (Phase Autofocus).

    Раздел 8.3.3 книги Оздемира (JTF-Based MOCOMP / Point-like target method).

    1. Находим доминирующий рассеиватель (бин с максимальной средней амплитудой).
    2. Извлекаем фазу этого бина для каждого импульса.
    3. Вычитаем эту фазу из всех бинов каждого импульса.

    Args:
        P_aligned: выровненная матрица профилей дальности M×N.
                   Ось 0 — дальность, ось 1 — медленное время.
        range_axis: ось дальностей (M,). Если None, используется индекс.

    Returns:
        P_comp: компенсированная матрица M×N (комплексная, не модуль!).
        ref_bin: индекс опорного бина.
        phase_errors: извлечённые фазовые ошибки (N,).
    """
    M, N = P_aligned.shape

    mean_amplitude = np.mean(np.abs(P_aligned), axis=1)
    ref_bin = np.argmax(mean_amplitude)

    phase_errors = np.angle(P_aligned[ref_bin, :])

    correction = np.exp(-1j * phase_errors)

    P_comp = P_aligned.astype(np.complex128)
    for n in range(N):
        P_comp[:, n] = P_aligned[:, n] * correction[n]

    return P_comp, ref_bin, phase_errors


def mocomp(P, range_axis=None, align_kwargs=None):
    """Полный конвейер компенсации поступательного движения (MOCOMP).

    Шаг 1: Выравнивание огибающей (Envelope Alignment).
    Шаг 2: Коррекция фазы (Phase Autofocus).

    Args:
        P: матрица профилей дальности M×N (модуль IFFT, fftshift).
           Ось 0 — дальность, ось 1 — медленное время.
        range_axis: ось дальностей (м). Если None — не используется.
        align_kwargs: dict, пробрасывается в envelope_alignment
            (None → дефолты v2: subbin=True, ref_mode='average').

    Returns:
        P_comp: компенсированная матрица M×N (комплексная).
        shifts: массив сдвигов огибающей (N,) float — в бинах.
        ref_bin: индекс опорного бина.
        phase_errors: извлечённые фазовые ошибки (N,).
    """
    kwargs = align_kwargs if align_kwargs is not None else {}
    P_aligned, shifts = envelope_alignment(P, **kwargs)
    P_comp, ref_bin, phase_errors = phase_autofocus(P_aligned, range_axis)
    return P_comp, shifts, ref_bin, phase_errors