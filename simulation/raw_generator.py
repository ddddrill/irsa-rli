import math
from dataclasses import dataclass, field

import numpy as np

from simulation.rng import spawn_rng
from simulation.scatterer_effects import (
    anisotropy_weights,
    make_complex_amplitudes,
    make_scatterer_patterns,
    occlusion_mask,
)
from simulation.system_effects import phase_noise_process, two_way_antenna_weights

SPEED_OF_LIGHT = 3e8


@dataclass
class SimEffects:
    """Параметры эффектов реализма. Все флаги по умолчанию ВЫКЛЮЧЕНЫ —
    в этом режиме генератор работает по классической формуле
    E[m,n] = sum_i sigma_i * exp(-j*4*pi*f_m*R_i(t_n)/c).

    Атрибуты:
        seed: базовый seed генераторов случайности.
        use_complex_sigma: случайные фазы рассеивателей sigma_i*exp(j*phi_i).
        use_anisotropy: гауссова угловая зависимость sigma_i(theta).
        anisotropy_width_deg: ширина лепестка анизотропии (град).
        use_shadowing: геометрическое затенение рассеивателей.
        shadow_radius_m: радиус затенения по поперечнику (м).
        use_antenna_pattern: двухпутевая ДН антенны G^2 на импульс.
        pointing_error_deg: ошибка наведения (град).
        antenna_drift_deg: дрейф отклонения луча за апертуру (град).
        use_phase_noise: фазовый шум опорного генератора (винер).
        phase_noise_std_deg: RMS фазового набега за апертуру (град).
    """

    seed: int = 42
    use_complex_sigma: bool = False
    use_anisotropy: bool = False
    anisotropy_width_deg: float = 10.0
    use_shadowing: bool = False
    shadow_radius_m: float = 0.4
    use_antenna_pattern: bool = False
    pointing_error_deg: float = 0.0
    antenna_drift_deg: float = 0.0
    use_phase_noise: bool = False
    phase_noise_std_deg: float = 10.0

    @property
    def any_enabled(self) -> bool:
        """True, если хотя бы один эффект реализма включён."""
        return any([
            self.use_complex_sigma,
            self.use_anisotropy,
            self.use_shadowing,
            self.use_antenna_pattern,
            self.use_phase_noise,
        ])


def add_complex_awgn(E_clean, snr_db, rng=None):
    """Добавить комплексный аддитивный белый гауссовский шум (Complex AWGN).

    Физика шума:
        - Вещественная и мнимая части шума — независимые N(0,1)
        - Мощность комплексного шума = сумма мощностей Re и Im частей

    Args:
        E_clean: чистая комплексная матрица M×N.
        snr_db: отношение сигнал/шум в децибелах.
                 np.inf — вернуть чистый сигнал без шума.
        rng: источник случайности (np.random.Generator) из spawn_rng(seed,
             "awgn") — даёт побитовую воспроизводимость при любом SNR.
             None — np.random.default_rng() (только для внешних вызовов;
             основной путь — generate_raw_matrix передаёт seeded-генератор).

    Returns:
        E_noisy: зашумленная матрица M×N.
    """
    if np.isinf(snr_db) or snr_db > 1000:
        return E_clean

    P_signal = np.mean(np.abs(E_clean) ** 2)

    SNR_lin = 10.0 ** (snr_db / 10.0)
    P_noise = P_signal / SNR_lin

    M, N = E_clean.shape
    if rng is None:
        rng = np.random.default_rng()
    N_raw = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))

    N_scaled = np.sqrt(P_noise / 2.0) * N_raw

    return E_clean + N_scaled


def generate_raw_matrix_loop(radar, target):
    """Сгенерировать матрицу сырых данных SFCW вложенными циклами (черновик).

    .. deprecated:: 2.4.0
        Учебная эталонная реализация, оставлена для проверки векторизованной
        версии. Не поддерживает эффекты реализма.

    Ось 0 (строки)  — быстрое время (частоты f_m), M штук.
    Ось 1 (столбцы) — медленное время (импульсы n), N штук.

    Формула заполнения:
        E[m, n] = sum_i  intens_i * exp( -j * 4π * f_m / c * R_i(t_n) )
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


def generate_raw_matrix(radar, target, effects=None):
    """Сгенерировать матрицу сырых данных SFCW (векторизованная версия).

    Цепочка применения эффектов реализма (все опциональны, по умолчанию
    выключены — классическая точечная модель с вещественными амплитудами):

        1. Комплексные амплитуды: sigma_i -> sigma_i * exp(j*phi_i)
        2. Анизотропия: W_aniso[n, i] = exp(-(theta_n - theta_0i)^2 / 2*theta_c^2)
        3. Затенение: visible[n, i] = 0, если точка перекрыта другой
        4. Суммирование эха: E[m, n] = sum_i sigma_i * W[n, i] * exp(j*phi_mn)
        5. Антенна: E[:, n] *= G^2(theta_off_n)
        6. Фазовый шум: E[:, n] *= exp(j*phi_noise_n)
        7. AWGN (как и раньше, по radar.snr_db)

    Broadcasting:
        f_r    -> shape (M, 1, 1)   — частоты (быстрое время)
        sigma  -> shape (1, 1, I)   — амплитуды рассеивателей
        W      -> shape (1, N, I)   — веса (анизотропия × затенение)
        R      -> shape (1, N, I)   — дальности (медленное время × рассеиватели)

    Args:
        radar: объект Radar (содержит snr_db).
        target: объект Target.
        effects: SimEffects или None (None/выключенные флаги = классическая
            модель, результат идентичен прежней версии).

    Returns:
        E: комплексная матрица M×N (зашумлена, если snr_db задан).
        f_r: вектор частот (M,).
    """
    eff = effects if effects is not None else SimEffects()

    _, _, f_r, _, _, _, _, _, _, _, _ = radar.base_img_params()
    M = len(f_r)
    N = target.num_pulses

    sc = target._load_scatterers()
    intens = sc["intens"]
    I = len(intens)

    R_all = target.get_ranges_all()          # (N, I)
    y_abs = target.get_cross_range_all()     # (N, I)

    # Амплитуды: вещественные (классика) или комплексные с фазами
    if eff.use_complex_sigma:
        sigma = make_complex_amplitudes(intens, eff.seed)
    else:
        sigma = intens[np.newaxis, np.newaxis, :].astype(np.complex128)

    # Веса рассеивателей (N, I): анизотропия × затенение
    W = np.ones((N, I), dtype=float)
    if eff.use_anisotropy or eff.use_shadowing:
        theta = np.rad2deg(target.get_aspect_angles())
        if eff.use_anisotropy:
            theta0 = make_scatterer_patterns(
                np.deg2rad(theta), I, eff.seed,
            )
            W *= anisotropy_weights(
                theta0, np.deg2rad(theta), math.radians(eff.anisotropy_width_deg),
            )
        if eff.use_shadowing:
            W *= occlusion_mask(R_all, y_abs, eff.shadow_radius_m)
        # W (N, I): broadcasting даёт sigma shape (N, I), совместимую с phi (M, N, I)
        sigma = sigma * W

    # Фаза распространения
    f_col = f_r[:, np.newaxis, np.newaxis]           # (M, 1, 1)
    R_row = R_all[np.newaxis, :, :]                  # (1, N, I)
    phi = -4.0 * np.pi * f_col / SPEED_OF_LIGHT * R_row

    E_clean = np.sum(sigma * np.exp(1j * phi), axis=2)   # (M, N)

    # Двухпутевая ДН антенны (вес по импульсам)
    if eff.use_antenna_pattern:
        ant = two_way_antenna_weights(
            radar.beam_width_deg(), N,
            pointing_error_deg=eff.pointing_error_deg,
            drift_deg=eff.antenna_drift_deg,
        )
        E_clean = E_clean * ant[np.newaxis, :]

    # Фазовый шум опорного генератора (по импульсам)
    if eff.use_phase_noise:
        phn = phase_noise_process(N, eff.phase_noise_std_deg, eff.seed)
        E_clean = E_clean * np.exp(1j * phn)[np.newaxis, :]

    # Шум из seeded-подгенератора "awgn" (изоляция от других эффектов,
    # побитовая воспроизводимость при повторном прогоне с тем же seed).
    E_noisy = add_complex_awgn(E_clean, radar.snr_db,
                               rng=spawn_rng(eff.seed, "awgn"))

    return E_noisy, f_r
