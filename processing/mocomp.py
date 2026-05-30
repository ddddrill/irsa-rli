import numpy as np
from scipy.signal import correlate


def envelope_alignment(P, method="correlation"):
    """Выравнивание огибающей профилей дальности (Envelope Alignment).

    Раздел 8.3.1 книги Оздемира (Cross-Correlation Method).

    Цель: сдвинуть каждый столбец матрицы P так, чтобы пики от цели
    выстроились в горизонтальную линию (убрать Range Walk).

    Args:
        P: матрица профилей дальности M×N (модуль IFFT, fftshift).
           Ось 0 — дальность, ось 1 — медленное время (импульсы).
        method: метод вычисления сдвига ("correlation" или "centroid").

    Returns:
        P_aligned: выровненная матрица M×N.
        shifts: массив сдвигов (N,) в отсчетах.
    """
    M, N = P.shape
    P_aligned = np.zeros_like(P)
    shifts = np.zeros(N, dtype=int)

    ref = P[:, 0]
    P_aligned[:, 0] = ref

    for n in range(1, N):
        current = P[:, n]

        if method == "correlation":
            corr = correlate(current, ref, mode="full")
            lag = np.argmax(corr) - (M - 1)
            shifts[n] = lag
            P_aligned[:, n] = np.roll(current, -lag)

        elif method == "centroid":
            ref_centroid = np.average(np.arange(M), weights=ref ** 2 + 1e-30)
            cur_centroid = np.average(np.arange(M), weights=current ** 2 + 1e-30)
            lag = int(round(cur_centroid - ref_centroid))
            shifts[n] = lag
            P_aligned[:, n] = np.roll(current, -lag)

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

    mean_amplitude = np.mean(P_aligned, axis=1)
    ref_bin = np.argmax(mean_amplitude)

    phase_errors = np.angle(P_aligned[ref_bin, :])

    correction = np.exp(-1j * phase_errors)

    P_comp = P_aligned.astype(np.complex128)
    for n in range(N):
        P_comp[:, n] = P_aligned[:, n] * correction[n]

    return P_comp, ref_bin, phase_errors


def mocomp(P, range_axis=None):
    """Полный конвейер компенсации поступательного движения (MOCOMP).

    Шаг 1: Выравнивание огибающей (Envelope Alignment).
    Шаг 2: Коррекция фазы (Phase Autofocus).

    Args:
        P: матрица профилей дальности M×N (модуль IFFT, fftshift).
           Ось 0 — дальность, ось 1 — медленное время.
        range_axis: ось дальностей (м). Если None — не используется.

    Returns:
        P_comp: компенсированная матрица M×N (комплексная).
        shifts: массив сдвигов огибающей (N,).
        ref_bin: индекс опорного бина.
        phase_errors: извлечённые фазовые ошибки (N,).
    """
    P_aligned, shifts = envelope_alignment(P, method="correlation")
    P_comp, ref_bin, phase_errors = phase_autofocus(P_aligned, range_axis)
    return P_comp, shifts, ref_bin, phase_errors
