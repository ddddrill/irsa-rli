"""Эффекты реализма на уровне рассеивателей.

Физика:
    1. Комплексные амплитуды. У реального центра рассеяния амплитуда и фаза
       зависят от локальной геометрии (материала, кромок, шероховатости).
       В точечной модели это сводится к случайной фазе phi_i ~ U(0, 2*pi):
       sigma_i -> sigma_i * exp(j*phi_i).  (Swerling 0 -> "phase-only")

    2. Анизотропия. Отражатель «светится» не равномерно по ракурсу, а в
       пределах некоторой ширины theta_c (диэдры и кромки дают узкие
       пики, сферы — широкие). Модель — гауссова лепестковая ДН:
           sigma_i(theta_n) = sigma_i * exp(-(theta_n - theta_0i)^2 / (2*theta_c^2))
       где theta_0i — ракурс максимального отражения i-й точки.

    3. Затенение (occlusion). Точка на обратной стороне панели отражает
       хуже или не отражает вовсе. Геометрическое приближение: рассеиватель
       i гасится на импульсе n, если между ним и радаром по линии визия
       оказывается другой рассеиватель j (ближе по дальности, с поперечным
       смещением меньше shadow_radius).

Все функции чистые и векторизованные; случайность — через переданный RNG.
"""
import numpy as np

from simulation.rng import spawn_rng


def make_complex_amplitudes(intens, seed, rng=None):
    """Комплексные амплитуды рассеивателей: sigma_i * exp(j*phi_i).

    Args:
        intens: вещественные интенсивности, shape (I,).
        seed: базовый seed конфигурации.
        rng: готовый генератор (перекрывает seed).

    Returns:
        sigma_complex: комплексный массив shape (I,), |sigma| = intens.
    """
    rng = rng if rng is not None else spawn_rng(seed, "complex_sigma")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(intens))
    return np.asarray(intens, dtype=np.complex128) * np.exp(1j * phases)


def make_scatterer_patterns(theta, num_scatterers, seed, rng=None):
    """Случайные пики анизотропии theta_0i в пределах сектора съёмки.

    Пики равномерно распределены по сектору с запасом ±25% ширины сектора:
    часть рассеивателей достигает максимума отражения за пределами
    наблюдаемого ракурса (виден только «склон» лепестка).

    Args:
        theta: ось углов наблюдения, shape (N,), рад.
        num_scatterers: число рассеивателей I.
        seed: базовый seed конфигурации.
        rng: готовый генератор (перекрывает seed).

    Returns:
        theta0: пиковые ракурсы, shape (I,), рад.
    """
    rng = rng if rng is not None else spawn_rng(seed, "anisotropy")
    lo, hi = float(np.min(theta)), float(np.max(theta))
    width = hi - lo
    return rng.uniform(lo - 0.25 * width, hi + 0.25 * width, size=num_scatterers)


def anisotropy_weights(theta0, theta, theta_c):
    """Гауссовы весовые коэффициенты анизотропии W[i, n].

    Формула:
        W[i, n] = exp(-(theta_n - theta_0i)^2 / (2 * theta_c^2))

    Args:
        theta0: пиковые ракурсы рассеивателей, shape (I,), рад.
        theta: углы наблюдения, shape (N,), рад.
        theta_c: ширина лепестка, рад (больше 0).

    Returns:
        W: весовая матрица shape (N, I), значения (0, 1].
    """
    theta = np.asarray(theta, dtype=float)[np.newaxis, :]      # (1, N)
    theta0 = np.asarray(theta0, dtype=float)[:, np.newaxis]    # (I, 1)
    theta_c = max(float(theta_c), 1e-9)
    W = np.exp(-((theta - theta0) ** 2) / (2.0 * theta_c ** 2))
    return W.T  # (N, I) — согласованная с матрицей дальностей форма


def occlusion_mask(ranges, y_abs, shadow_radius):
    """Маска видимости рассеивателей по геометрическому затенению.

    Точка i затенена на импульсе n, если существует точка j:
        R_j < R_i  (ближе к радару)  и  |y_j - y_i| < shadow_radius
        (та же «труба» по поперечному направлению).

    Args:
        ranges: матрица дальностей shape (N, I).
        y_abs: поперечные координаты рассеивателей shape (N, I).
        shadow_radius: радиус затенения, м.

    Returns:
        visible: булева маска shape (N, I). True = виден.
    """
    N, I = ranges.shape
    R = np.asarray(ranges, dtype=float)
    Y = np.asarray(y_abs, dtype=float)

    # Парные сравнения: dist[i, j, n] и поперечное перекрытие
    # Для экономии памяти считаем попарно по кадрам (I обычно < 300).
    visible = np.ones((N, I), dtype=bool)
    for n in range(N):
        Rn, Yn = R[n], Y[n]
        closer = Rn[np.newaxis, :] < Rn[:, np.newaxis]           # (I, I): R_j < R_i
        overlap = np.abs(Yn[np.newaxis, :] - Yn[:, np.newaxis]) < shadow_radius
        blocked = np.any(closer & overlap, axis=1)
        visible[n] = ~blocked
    return visible
