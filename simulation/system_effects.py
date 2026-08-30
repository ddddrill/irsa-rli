"""Системные эффекты радиотракта: фазовый шум и антенна.

Физика:
    1. Фазовый шум опорного генератора. Двухточечная модель: наборег фазы
       за время наблюдения — винеровский процесс (интеграл белого шума),
       общий для всех рассеивателей и частотных шагов одного импульса.
       Это доминирующий механизм потери когерентности при секундных
       накоплениях (пассивная когерентность LEO-наблюдений). Модель
       нормирована: RMS фазового набега за апертуру равен заданному
       phase_std_deg.

    2. Двухпутевая диаграмма направленности. Амплитуда эха на импульсе n
       взвешивается на G^2(theta_off_n), где G — однопутевая гауссова
       ДН с шириной beam_width по уровню -3 дБ, а theta_off_n —
       угловое отклонение цели от луча (ошибка наведения + дрейф).
       Угловой размер спутника с 500 км << ДН, поэтому вес общий для
       всей цели; покомпонентная зависимость от ракурса — в
       scatterer_effects.anisotropy_weights.
"""
import numpy as np

from simulation.rng import spawn_rng


def phase_noise_process(num_pulses, phase_std_deg, seed, rng=None):
    """Винеровский фазовый процесс с заданным RMS за апертуру.

    phi(t_n) = cumsum(w_n) * sqrt(dt), w_n ~ N(0, 1)
    затем линейный тренд удалён (собственные колебания осциллятора не
    дают систематического тренда) и процесс отнормирован так, что
    std(phi) = phase_std_deg.

    Args:
        num_pulses: число импульсов N.
        phase_std_deg: RMS фазового набега за апертуру, град.
        seed: базовый seed конфигурации.
        rng: готовый генератор (перекрывает seed).

    Returns:
        phase: фазовый процесс, shape (N,), рад.
    """
    rng = rng if rng is not None else spawn_rng(seed, "phase_noise")
    w = rng.standard_normal(num_pulses)
    phi = np.cumsum(w)
    phi = phi - np.linspace(phi[0], phi[-1], num_pulses)  # убрать тренд
    std = np.std(phi)
    if std > 1e-12:
        phi = phi / std
    target_std_rad = np.deg2rad(phase_std_deg)
    return phi * target_std_rad


def two_way_antenna_weights(beam_width_deg, num_pulses, pointing_error_deg=0.0,
                            drift_deg=0.0):
    """Вес эха G^2(theta_off_n) по двухпутевой гауссовой ДН.

    Гауссова ДН по уровню -3 дБ:
        G(theta) = exp(-2.7726 * (theta / beam_width)^2)   [однопутевая]
    Двухпутевая (передача + приём) — квадрат:
        W(theta) = G(theta)^2 = exp(-5.545 * (theta/beam_width)^2)

    Args:
        beam_width_deg: ширина ДН по -3 дБ, град.
        num_pulses: число импульсов N.
        pointing_error_deg: постоянная ошибка наведения, град.
        drift_deg: линейный дрейф отклонения за апертуру, град
                   (суммарное смещение от первого к последнему импульсу).

    Returns:
        weights: весовые коэффициенты, shape (N,), значения (0, 1].
    """
    n = np.arange(num_pulses, dtype=float)
    if num_pulses > 1:
        t = n / (num_pulses - 1.0)
    else:
        t = np.zeros(1)
    theta_off = pointing_error_deg + drift_deg * (t - 0.5)
    beam = max(float(beam_width_deg), 1e-9)
    return np.exp(-5.545 * (theta_off / beam) ** 2)
