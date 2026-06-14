import numpy as np
from scipy.fft import fft, fftshift

SPEED_OF_LIGHT = 3e8


def _hamming_window(N):
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N) / (N - 1))


def _hann_window(N):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1)))


def _boxcar_window(N):
    return np.ones(N)


WINDOW_FUNCTIONS = {
    "hamming": _hamming_window,
    "hann": _hann_window,
    "none": _boxcar_window,
    None: _boxcar_window,
}


def azimuth_compress(P_comp, omega, f_c, pri=1.0, window="none"):
    """Азимутальное сжатие: 1D FFT по оси медленного времени.

    Раздел 6 книги Оздемира. Переход из временной области (импульсы)
    в частотную (доплер/азимут).

    Алгоритм:
        1. (Опционально) Взвешивание оконной функцией по оси axis=1.
        2. FFT по axis=1 (по столбцам — медленное время).
        3. fftshift по axis=1 (центрирование спектра).

    Args:
        P_comp: компенсированная матрица M×N (ось 0 = дальность, ось 1 = импульсы).
        omega: угловая скорость вращения цели (рад/с).
        f_c: центральная частота радара (Гц).
        pri: период повторения импульсов (с).
        window: тип оконной функции ("hamming", "hann", "none").

    Returns:
        I: комплексная матрица изображения M×N (ось 0 = дальность, ось 1 = азимут).
        azimuth_axis: ось азимута в метрах (N,).
        doppler_axis: ось доплеровских частот (Нц) (N,).
    """
    M, N = P_comp.shape

    win = WINDOW_FUNCTIONS.get(window, _boxcar_window)(N)
    P_windowed = P_comp * win[np.newaxis, :]

    I = fftshift(fft(P_windowed, axis=1), axes=1)

    T_obs = N * pri
    df_doppler = 1.0 / T_obs
    doppler_axis = (np.arange(N) - N // 2) * df_doppler

    lam = SPEED_OF_LIGHT / f_c
    if omega != 0.0:
        dx = lam / (2.0 * omega * T_obs)
    else:
        dx = 1.0
    X_size = N * dx
    azimuth_axis = np.linspace(-X_size / 2.0, X_size / 2.0, N)

    return I, azimuth_axis, doppler_axis


def compute_amplitude(I, mode="linear", db_floor=-40.0, normalize=True):
    """Вычислить амплитуду изображения для отображения.

    Args:
        I: комплексная матрица изображения.
        mode: "linear" (|I|) или "dB" / "db" (20*log10(|I|)).
        db_floor: нижняя граница дБ (если mode="dB").
        normalize: если True и mode="dB", нормализовать так, чтобы max = 0 dB.

    Returns:
        amplitude: вещественная матрица амплитуд.
    """
    amp = np.abs(I)
    if mode.lower() == "db":
        amp_db = 20.0 * np.log10(amp + 1e-30)
        if normalize:
            amp_max = amp_db.max()
            if amp_max > 0:
                amp_db = amp_db - amp_max
        amp_db = np.maximum(amp_db, db_floor)
        return amp_db
    return amp
