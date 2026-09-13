import numpy as np
from scipy.fft import ifft, fftshift

from processing.windows import get_weighting

SPEED_OF_LIGHT = 3e8


def range_compress(E, f_r, R0=0.0, window="none", window_param=None):
    """Сжатие по дальности: 1D IFFT по оси частот (быстрое время).

    Переход из частотной области (сырые данные SFCW) в дальностную область
    (профили дальности). Раздел 6.5 книги Оздемира.

    Если R0 != 0, предварительно выполняется дешифровка (dechirp):
    умножение на опорный сигнал exp(+j*4π*f*R0/c), чтобы вычесть фазу
    начальной дальности. После дешифрки цель появляется в бине,
    соответствующем дифференциальной дальности (R - R0).

    Опциональное взвешивание применяется к столбцам E (axis=0) ДО ifft —
    C&M, §3.3: окно управляет боковыми лепестками только до преобразования
    Фурье. Когерентное усиление компенсируется нормировкой win/win.mean()
    (C&M, §6.3.1), так что амплитуда пиков сопоставима между окнами.
    Для window="none" (boxcar) нормировка не применяется — результат
    битово идентичен версии без взвешивания.

    Args:
        E: комплексная матрица сырых данных M×N (частоты × импульсы).
        f_r: вектор частот (M,).
        R0: начальная дальность до ЦМ (м). Если 0 — дешифровка не нужна.
        window: окно взвешивания: "none" | "hamming" | "hann" | "kaiser" |
            "chebwin" | "taylor".
        window_param: beta для kaiser (по умолч. 8.0) или at-дБ для chebwin
            (по умолч. 60.0); для taylor — dict(nbar=4, sll=-35.0).

    Returns:
        P: матрица профилей дальности M×N (комплексная, fftshift).
        P_abs: модуль |P| (вещественная, для визуализации и корреляции).
        range_axis: ось дальностей в метрах (M,). Нулевая точка = R0.
        dr: разрешение по дальности (м).
    """
    M, N = E.shape

    if R0 != 0.0:
        E_ref = np.exp(1j * 4.0 * np.pi * f_r / SPEED_OF_LIGHT * R0)
        E = E * E_ref[:, np.newaxis]

    win = get_weighting(window, M, window_param)
    E = E * win[:, np.newaxis]

    P = fftshift(ifft(E, axis=0), axes=0)
    P_abs = np.abs(P)

    B = f_r[-1] - f_r[0]
    df = B / (M - 1) if M > 1 else B / M
    dr = SPEED_OF_LIGHT / (2.0 * M * df)

    range_axis = (np.arange(M) - (M - 1) / 2.0) * dr

    return P, P_abs, range_axis, dr
