import numpy as np
from scipy.fft import ifft, fftshift

SPEED_OF_LIGHT = 3e8


def range_compress(E, f_r, R0=0.0):
    """Сжатие по дальности: 1D IFFT по оси частот (быстрое время).

    Переход из частотной области (сырые данные SFCW) в дальностную область
    (профили дальности). Раздел 6.5 книги Оздемира.

    Если R0 != 0, предварительно выполняется дешифровка (dechirp):
    умножение на опорный сигнал exp(+j*4π*f*R0/c), чтобы вычесть фазу
    начальной дальности. После дешифрки цель появляется в бине,
    соответствующем дифференциальной дальности (R - R0).

    Args:
        E: комплексная матрица сырых данных M×N (частоты × импульсы).
        f_r: вектор частот (M,).
        R0: начальная дальность до ЦМ (м). Если 0 — дешифровка не нужна.

    Returns:
        P: матрица профилей дальности M×N (модуль IFFT, fftshift).
        range_axis: ось дальностей в метрах (M,). Нулевая точка = R0.
        dr: разрешение по дальности (м).
    """
    M, N = E.shape

    if R0 != 0.0:
        E_ref = np.exp(1j * 4.0 * np.pi * f_r / SPEED_OF_LIGHT * R0)
        E = E * E_ref[:, np.newaxis]

    P_raw = ifft(E, axis=0)
    P = np.abs(fftshift(P_raw, axes=0))

    B = f_r[-1] - f_r[0]
    df = B / (M - 1) if M > 1 else B / M
    dr = SPEED_OF_LIGHT / (2.0 * M * df)

    range_axis = np.arange(M) * dr
    range_axis = range_axis - range_axis[M - 1] / 2.0

    return P, range_axis, dr
