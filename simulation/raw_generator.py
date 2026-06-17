import numpy as np

SPEED_OF_LIGHT = 3e8


def add_complex_awgn(E_clean, snr_db):
    """Добавить комплексный аддитивный белый гауссовский шум (Complex AWGN).

    Физика шума:
        - Вещественная и мнимая части шума — независимые N(0,1)
        - Мощность комплексного шума = сумма мощностей Re и Im частей

    Args:
        E_clean: чистая комплексная матрица M×N.
        snr_db: отношение сигнал/шум в децибелах.
                 np.inf — вернуть чистый сигнал без шума.

    Returns:
        E_noisy: зашумленная матрица M×N.
    """
    if np.isinf(snr_db) or snr_db > 1000:
        return E_clean

    P_signal = np.mean(np.abs(E_clean) ** 2)

    SNR_lin = 10.0 ** (snr_db / 10.0)
    P_noise = P_signal / SNR_lin

    M, N = E_clean.shape
    N_raw = np.random.randn(M, N) + 1j * np.random.randn(M, N)

    N_scaled = np.sqrt(P_noise / 2.0) * N_raw

    return E_clean + N_scaled


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
        radar: объект Radar (содержит snr_db).
        target: объект Target.

    Returns:
        E: комплексная матрица M×N (зашумлена, если snr_db задан).
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

    E_clean = np.sum(s_col * np.exp(1j * phi), axis=2)

    E_noisy = add_complex_awgn(E_clean, radar.snr_db)

    return E_noisy, f_r
