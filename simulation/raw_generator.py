import numpy as np

SPEED_OF_LIGHT = 3e8


def generate_raw_matrix_loop(radar, target):
    """Сгенерировать матрицу сырых данных SFCW вложенными циклами (черновик).

    Ось 0 (строки)  — быстрое время (частоты f_m), M штук.
    Ось 1 (столбцы) — медленное время (импульсы n), N штук.

    Формула заполнения:
        E[m, n] = sum_i  intens_i * exp( -j * 4π * f_m / c * R_i(t_n) )

    Args:
        radar: объект Radar (models/radar.py) с параметрами сигнала.
        target: объект Target (models/target.py) с кинематикой цели.

    Returns:
        E: комплексная матрица M×N.
        f_r: вектор частот (M,).
    """
    _, _, f_r, _, _, _, _, _, _, _, _ = radar.base_img_params()
    M = len(f_r)
    N = target.num_pulses

    sc = target._load_scatterers()
    intens = sc["intens"]
    I = len(intens)

    E = np.zeros((M, N), dtype="complex128")

    for n in range(N):
        t_n = target.slow_time(n)
        x_abs, y_abs = target._absolute_positions(t_n, sc["x0"], sc["y0"])
        R = np.sqrt(x_abs ** 2 + y_abs ** 2)

        for m in range(M):
            f_m = f_r[m]
            for i in range(I):
                phi = -4.0 * np.pi * f_m / SPEED_OF_LIGHT * R[i]
                E[m, n] += intens[i] * np.exp(1j * phi)

    return E, f_r


def generate_raw_matrix(radar, target):
    """Сгенерировать матрицу сырых данных SFCW (векторизованная версия).

    Использует NumPy broadcasting вместо вложенных циклов Python.

    Матрица фаз:
        phi[i, n] = -4π * f_m / c * R_i(t_n)

    Broadcasting:
        f_r  → shape (M, 1, 1)   — частоты (быстрое время)
        R    → shape (1, N, I)   — дальности (медленное время × рассеиватели)
        intens → shape (1, 1, I) — ЭПР рассеивателей

    E[m, n] = sum_i intens_i * exp(j * phi[i, n])

    Args:
        radar: объект Radar.
        target: объект Target.

    Returns:
        E: комплексная матрица M×N.
        f_r: вектор частот (M,).
    """
    _, _, f_r, _, _, _, _, _, _, _, _ = radar.base_img_params()
    M = len(f_r)
    N = target.num_pulses

    sc = target._load_scatterers()
    intens = sc["intens"]
    I = len(intens)

    R_all = target.get_ranges_all()

    f_col = f_r[:, np.newaxis, np.newaxis]
    R_row = R_all[np.newaxis, :, :]
    s_col = intens[np.newaxis, np.newaxis, :]

    phi = -4.0 * np.pi * f_col / SPEED_OF_LIGHT * R_row

    E = np.sum(s_col * np.exp(1j * phi), axis=2)

    return E, f_r
