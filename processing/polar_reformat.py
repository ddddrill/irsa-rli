"""Polar Reformatting: переход от полярной сетки к декартовой в k-space.

Реализация по книге Оздемира, раздел 4.6.

Физическая суть:
    Данные SFCW радара E[m, n] лежат на полярной сетке (k, theta).
    Алгоритм 2D БПФ требует равномерную декартову сетку (kx, ky).
    Этап 2: математическое осмысление k-space (без интерполяции данных).
    Этап 3: проектирование целевой декартовой сетки.
    Этап 4: интерполяция с полярной на декартову сетку.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import math

SPEED_OF_LIGHT = 3e8


def compute_wavenumber_axis(f_r):
    """Шаг 2.2: Формирование оси волновых чисел дальности k.

    Физический смысл:
        Волновое число k описывает пространственную частоту волны.
        Множитель 4*pi (а не 2*pi) появляется из-за двойного пути:
        волна идёт от радара до цели и обратно.

    Формула (Оздемир, 4.40):
        k_m = 4 * pi * f_m / c

    Args:
        f_r: вектор частот, shape (M,). Частоты каждого шага (Гц).

    Returns:
        k: вектор волновых чисел, shape (M,). Единицы: рад/м.
        k_min: минимальное волновое число.
        k_max: максимальное волновое число.
    """
    k = 4.0 * np.pi * f_r / SPEED_OF_LIGHT
    return k, k.min(), k.max()


def compute_angle_axis(omega, pri, num_pulses):
    """Шаг 2.3: Формирование оси углов наблюдения theta.

    Физический смысл:
        Каждый импульс n излучается в момент t_n = n * PRI.
        За это время цель поворачивается на угол theta_n = omega * t_n.
        Это угол наблюдения цели (азимутальный угол в k-space).

    Формула:
        theta_n = omega * n * PRI

    Args:
        omega: угловая скорость вращения цели (рад/с).
        pri: период повторения импульсов (с).
        num_pulses: число импульсов N.

    Returns:
        theta: вектор углов, shape (N,). Единицы: рад.
        delta_theta: суммарный угол поворота (рад).
    """
    n = np.arange(num_pulses)
    theta = omega * n * pri
    delta_theta = theta[-1] - theta[0]
    return theta, delta_theta


def polar_to_cartesian(k, theta):
    """Шаг 2.4: Переход от полярных координат (k, theta) к декартовым (kx, ky).

    Физический смысл:
        Kx — ось волновых чисел дальности (range wavenumber).
        Ky — ось волновых чисел азимута (cross-range wavenumber).

    Формулы (Оздемир, 4.41):
        Kx[m, n] = k_m * cos(theta_n)
        Ky[m, n] = k_m * sin(theta_n)

    Args:
        k: вектор волновых чисел, shape (M,).
        theta: вектор углов, shape (N,).

    Returns:
        Kx: двумерный массив, shape (M, N). Ось kx.
        Ky: двумерный массив, shape (M, N). Ось ky.
    """
    Kx = np.outer(k, np.cos(theta))
    Ky = np.outer(k, np.sin(theta))
    return Kx, Ky


def get_kspace_bounds(k, theta):
    """Рассчитать границы кольцевого сектора в декартовых координатах.

    Физический смысл:
        Описывает прямоугольник, в который вписывается кольцевой сектор.
        Используется для проектирования целевой декартовой сетки (Этап 3).

    Формулы (Оздемир, 4.55-4.56):
        Kx_min = k_min * cos(theta_max)
        Kx_max = k_max
        Ky_min = -k_max * sin(theta_max)
        Ky_max = k_max * sin(theta_max)

    Args:
        k: вектор волновых чисел, shape (M,).
        theta: вектор углов, shape (N,).

    Returns:
        dict с границами: kx_min, kx_max, ky_min, ky_max, k_min, k_max, theta_max.
    """
    k_min, k_max = k.min(), k.max()
    theta_max = np.abs(theta).max()

    kx_min = k_min * np.cos(theta_max)
    kx_max = k_max
    ky_min = -k_max * np.sin(theta_max)
    ky_max = k_max * np.sin(theta_max)

    return {
        "kx_min": kx_min,
        "kx_max": kx_max,
        "ky_min": ky_min,
        "ky_max": ky_max,
        "k_min": k_min,
        "k_max": k_max,
        "theta_max": theta_max,
    }


def plot_kspace(Kx, Ky, E, ax=None, title=None, show_rectangle=True):
    """Шаг 2.5: Визуализация кольцевого сектора в k-space.

    Строит 2D карту распределения энергии в пространстве волновых чисел.
    При малых углах (~прямоугольник): Kx ~ k, Ky ~ k*theta.
    При больших углах (~сектор): видна кривизна полярной сетки.

    Args:
        Kx: массив kx координат, shape (M, N).
        Ky: массив ky координат, shape (M, N).
        E: матрица эхо-сигнала, shape (M, N). Комплексная.
        ax: matplotlib ось (если None — создаст свою фигуру).
        title: заголовок графика.
        show_rectangle: показать ожидаемый прямоугольник (для сравнения).

    Returns:
        ax: matplotlib ось с графиком.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    amp = np.abs(E)
    im = ax.pcolormesh(Kx, Ky, amp, cmap="plasma", shading="auto")
    ax.set_xlabel("Kx (rad/m)")
    ax.set_ylabel("Ky (rad/m)")
    ax.set_title(title or "K-space: annular sector")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|E|")

    if show_rectangle:
        kx_min, kx_max = Kx.min(), Kx.max()
        ky_min, ky_max = Ky.min(), Ky.max()
        rect_x = [kx_min, kx_max, kx_max, kx_min, kx_min]
        rect_y = [ky_min, ky_min, ky_max, ky_max, ky_min]
        ax.plot(rect_x, rect_y, "r--", linewidth=1, label="Bounding box")
        ax.legend(fontsize=8)

    return ax


def build_kspace_grid(f_r, omega, pri, num_pulses):
    """Полная сборка k-space: от сырых параметров к полярной сетке.

    Объединяет Шаги 2.2-2.4 в один вызов.

    Args:
        f_r: вектор частот, shape (M,).
        omega: угловая скорость (рад/с).
        pri: период повторения импульсов (с).
        num_pulses: число импульсов N.

    Returns:
        dict с ключами:
            k: вектор волновых чисел (M,)
            theta: вектор углов (N,)
            Kx: декартова ось kx (M, N)
            Ky: декартова ось ky (M, N)
            k_min, k_max: границы волновых чисел
            delta_theta: суммарный угол (рад)
            bounds: dict с границами сектора
    """
    k, k_min, k_max = compute_wavenumber_axis(f_r)
    theta, delta_theta = compute_angle_axis(omega, pri, num_pulses)
    Kx, Ky = polar_to_cartesian(k, theta)
    bounds = get_kspace_bounds(k, theta)

    return {
        "k": k,
        "theta": theta,
        "Kx": Kx,
        "Ky": Ky,
        "k_min": k_min,
        "k_max": k_max,
        "delta_theta": delta_theta,
        "bounds": bounds,
    }


# ── Этап 3: Проектирование декартовой сетки ────────────────────────────────


def design_cartesian_grid(k, theta, Xmax_scene, Ymax_scene):
    """Шаг 3.1-3.4: Проектирование целевой декартовой сетки.

    Физический смысл:
        Шаг сетки в k-пространстве определяет максимальный размер сцены
        в пространственной области (теорема Котельникова).
        Δkx <= 2π / Xmax_scene  — исключает алиасинг по дальности.
        Δky <= 2π / Ymax_scene  — исключает алиасинг по азимуту.

    Формулы (Оздемир, 4.55-4.56):
        Δkx = 2π / Xmax_scene
        Δky = 2π / Ymax_scene
        M_new = ceil((kx_max - kx_min) / Δkx)
        N_new = ceil((ky_max - ky_min) / Δky)

    Args:
        k: вектор волновых чисел, shape (M,).
        theta: вектор углов, shape (N,).
        Xmax_scene: максимальный размер сцены по дальности (м).
        Ymax_scene: максимальный размер сцены по азимуту (м).

    Returns:
        dict с ключами:
            Kx_grid, Ky_grid: 2D сетки координат (M_new, N_new)
            dkx, dky: шаг сетки (рад/м)
            M_new, N_new: размерность целевой матрицы
            kx_axis, ky_axis: 1D оси сетки
            bounds: границы сетки (kx_min, kx_max, ky_min, ky_max)
    """
    bounds = get_kspace_bounds(k, theta)
    kx_min, kx_max = bounds["kx_min"], bounds["kx_max"]
    ky_min, ky_max = bounds["ky_min"], bounds["ky_max"]

    # Шаг 3.2: Шаг сетки по теореме Котельникова
    dkx = 2.0 * np.pi / Xmax_scene
    dky = 2.0 * np.pi / Ymax_scene

    # Шаг 3.3: Количество точек
    M_new = max(1, int(np.ceil((kx_max - kx_min) / dkx)))
    N_new = max(1, int(np.ceil((ky_max - ky_min) / dky)))

    # Округление до степеней двойки для эффективности БПФ
    M_new = max(256, int(2 ** np.ceil(np.log2(max(M_new, 1)))))
    N_new = max(256, int(2 ** np.ceil(np.log2(max(N_new, 1)))))

    # Шаг 3.4: Генерация осей и meshgrid
    kx_axis = np.linspace(kx_min, kx_max, M_new)
    ky_axis = np.linspace(ky_min, ky_max, N_new)
    Kx_grid, Ky_grid = np.meshgrid(kx_axis, ky_axis, indexing="ij")

    return {
        "Kx_grid": Kx_grid,
        "Ky_grid": Ky_grid,
        "dkx": dkx,
        "dky": dky,
        "M_new": M_new,
        "N_new": N_new,
        "kx_axis": kx_axis,
        "ky_axis": ky_axis,
        "bounds": {
            "kx_min": kx_min,
            "kx_max": kx_max,
            "ky_min": ky_min,
            "ky_max": ky_max,
        },
    }


def plot_grid_comparison(Kx_polar, Ky_polar, grid_info, ax=None):
    """Шаг 3.5: Визуальное сравнение полярной и декартовой сеток.

    Накладывает на один график:
        - Красные точки: исходный кольцевой сектор (полярная сетка)
        - Синие точки: целевая прямоугольная сетка

    Args:
        Kx_polar: массив kx координат полярной сетки, shape (M, N).
        Ky_polar: массив ky координат полярной сетки, shape (M, N).
        grid_info: dict из design_cartesian_grid().
        ax: matplotlib ось (если None — создаст свою фигуру).

    Returns:
        ax: matplotlib ось с графиком.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Красные точки: полярная сетка (подвыборка для скорости)
    step_polar = max(1, Kx_polar.size // 5000)
    ax.scatter(Kx_polar.ravel()[::step_polar], Ky_polar.ravel()[::step_polar],
               s=0.3, c="red", alpha=0.5, label="Polar grid (data)")

    # Синие точки: целевая декартова сетка (подвыборка)
    Kx_grid = grid_info["Kx_grid"]
    Ky_grid = grid_info["Ky_grid"]
    step_grid = max(1, Kx_grid.size // 5000)
    ax.scatter(Kx_grid.ravel()[::step_grid], Ky_grid.ravel()[::step_grid],
               s=0.3, c="blue", alpha=0.3, label="Cartesian grid (target)")

    ax.set_xlabel("Kx (rad/m)")
    ax.set_ylabel("Ky (rad/m)")
    ax.set_title(
        f"Grid comparison: polar ({Kx_polar.shape[0]}x{Kx_polar.shape[1]}) "
        f"vs Cartesian ({grid_info['M_new']}x{grid_info['N_new']})"
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    return ax


# ── Этап 4: Интерполяция с полярной на декартову сетку ─────────────────────


def interpolate_polar_to_cartesian(E, k, theta, grid_info, method="linear"):
    """Шаг 4.1-4.5: Интерполяция данных с полярной сетки на декартову.

    Физический смысл:
        Для каждой ячейки целевой декартовой сетки (kx_grid, ky_grid)
        находим соответствующие полярные координаты (k_target, theta_target)
        и вычисляем значение сигнала интерполяцией из исходной матрицы E.

    Алгоритм (Шаг 4.5 - векторизация через scipy):
        1. Создаем RegularGridInterpolator на исходных осях (k, theta).
        2. Для каждой точки целевой сетки вычисляем (k_target, theta_target).
        3. Интерполятор возвращает значения (zero-filled за пределами).

    Args:
        E: исходная матрица эхо-сигнала, shape (M, N). Комплексная.
        k: вектор волновых чисел, shape (M,).
        theta: вектор углов, shape (N,).
        grid_info: dict из design_cartesian_grid().
        method: "nearest" (Шаг 4.2) или "linear" (Шаг 4.3).

    Returns:
        E_cart: интерполированная матрица, shape (M_new, N_new). Комплексная.
    """
    Kx_grid = grid_info["Kx_grid"]
    Ky_grid = grid_info["Ky_grid"]

    # Шаг 4.1: Обратное преобразование координат
    k_target = np.sqrt(Kx_grid**2 + Ky_grid**2)
    theta_target = np.arctan2(Ky_grid, Kx_grid)

    # Шаг 4.5: Векторизация через scipy.interpolate.RegularGridInterpolator
    # Интерполятор строится на исходных осях (k, theta)
    interpolator_real = RegularGridInterpolator(
        (k, theta), E.real,
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )
    interpolator_imag = RegularGridInterpolator(
        (k, theta), E.imag,
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )

    # Подготовка точек для интерполяции: array of (k, theta) pairs
    points = np.column_stack([k_target.ravel(), theta_target.ravel()])

    # Шаг 4.4: Обработка слепых зон (automatic via fill_value=0)
    real_part = interpolator_real(points).reshape(Kx_grid.shape)
    imag_part = interpolator_imag(points).reshape(Kx_grid.shape)

    E_cart = real_part + 1j * imag_part

    return E_cart


def plot_kspace_cartesian(E_cart, grid_info, ax=None, title=None):
    """Шаг 4.6: Визуализация интерполированной матрицы E_cartesian в k-space.

    Показывает результат полярного переформатирования:
        - Идеальный прямоугольник данных на равномерной декартовой сетке.
        - Нулевые области (тьма) по углам — там данных не было.

    Args:
        E_cart: интерполированная матрица, shape (M_new, N_new). Комплексная.
        grid_info: dict из design_cartesian_grid().
        ax: matplotlib ось (если None — создаст свою фигуру).
        title: заголовок графика.

    Returns:
        ax: matplotlib ось с графиком.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    kx_axis = grid_info["kx_axis"]
    ky_axis = grid_info["ky_axis"]
    amp = np.abs(E_cart)

    extent = [ky_axis[0], ky_axis[-1], kx_axis[-1], kx_axis[0]]
    im = ax.imshow(amp, aspect="auto", cmap="plasma", extent=extent)
    ax.set_xlabel("Ky (rad/m)")
    ax.set_ylabel("Kx (rad/m)")
    ax.set_title(title or "K-space: Cartesian (after reformatting)")
    plt.colorbar(im, ax=ax, label="|E_cartesian|")

    return ax


# ── Этап 5: Обработка слепых зон и оконная функция ─────────────────────────


def create_data_mask(E_cart, threshold=0.0):
    """Шаг 5.2: Формирование бинарной маски присутствия данных.

    Физический смысл:
        Маска определяет, где в матрице E_cartesian находятся реальные данные
        (интерполированные из полярного сектора), а где — слепые зоны (нули).
        Используется для жесткого обнуления слепых зон после оконной функции.

    Args:
        E_cart: интерполированная матрица, shape (M_new, N_new). Комплексная.
        threshold: порог амплитуды для определения "данных" (по умолчанию 0).

    Returns:
        mask: бинарная матрица, shape (M_new, N_new). 1 = данные, 0 = слепая зона.
    """
    amp = np.abs(E_cart)
    mask = (amp > threshold).astype(np.float64)
    return mask


def taylor_window_1d(n, nbar=4, sll=-35.0):
    """Генерация 1D окна Тейлора.

    Окно Тейлора позволяет контролировать уровень боковых лепестков (SLL)
    при минимальном расширении главного лепестка. Стандарт в радарной обработке.

    Args:
        n: длина окна (количество отсчетов).
        nbar: число нулей в области боковых лепестков (по умолчанию 4).
        sll: желаемый уровень боковых лепестков в дБ (по умолчанию -35 дБ).

    Returns:
        w: массив окна, shape (n,). Значения от 0 до 1.
    """
    # Физический параметр A, связанный с уровнем боковых лепестков
    A = np.log(10) * np.abs(sll) / (2.0 * np.pi)

    # Нормированная частота нулей
    sigma = n / (n + 2.0 * A - 1.0)

    # Базовая функция
    nn = np.arange(n)

    # Вычисление Fourier series coefficients
    F = np.zeros(nbar)
    for i in range(nbar):
        num = 1.0
        for j in range(1, nbar):
            if i != j:
                num *= 1.0 - (float(i) ** 2) / (sigma ** 2 * (A ** 2 + (float(j) - 0.5) ** 2))
        denom = 1.0
        for j in range(1, nbar):
            if i != j:
                denom *= 1.0 - (float(i) / float(j)) ** 2
        if A > 0:
            denom *= 1.0 - float(i) ** 2 / (sigma * A) ** 2
        F[i] = num / denom if abs(denom) > 1e-15 else 0.0

    # Строим окно
    w = np.ones(n, dtype=float)
    for idx in range(n):
        x = (nn[idx] - n / 2.0 + 0.5) / n
        w_val = np.sinc(x)
        for i in range(1, nbar):
            if i < len(F):
                w_val += F[i] * np.cos(2.0 * np.pi * i * x)
        w[idx] = w_val

    # Нормализация
    w = w / w.max() if w.max() > 0 else w

    return w


def apply_window_2d(E_cart, window_type="taylor", **kwargs):
    """Шаг 5.3: Применение 2D оконной функции к матрице E_cartesian.

    Физический смысл:
        Окно сглаживает резкие границы данных в k-пространстве,
        подавляя боковые лепестки на финальном изображении.
        E_windowed = E_cart * W_2D

    Args:
        E_cart: интерполированная матрица, shape (M_new, N_new). Комплексная.
        window_type: тип окна ("taylor", "hamming", "hann").
        **kwargs: дополнительные параметры для окна Тейлора (nbar, sll).

    Returns:
        E_windowed: матрица после оконной функции, shape (M_new, N_new).
        W_2D: 2D окно, shape (M_new, N_new).
    """
    M_new, N_new = E_cart.shape

    if window_type == "taylor":
        nbar = kwargs.get("nbar", 4)
        sll = kwargs.get("sll", -35.0)
        wx = taylor_window_1d(M_new, nbar=nbar, sll=sll)
        wy = taylor_window_1d(N_new, nbar=nbar, sll=sll)
    elif window_type == "hamming":
        wx = np.hamming(M_new)
        wy = np.hamming(N_new)
    elif window_type == "hann":
        wx = np.hanning(M_new)
        wy = np.hanning(N_new)
    else:
        wx = np.ones(M_new)
        wy = np.ones(N_new)

    W_2D = np.outer(wx, wy)
    E_windowed = E_cart * W_2D

    return E_windowed, W_2D


def zero_pad_k_space(E_final, target_shape=None, factor=2):
    """Шаг 5.4: Дополнение нулями (Zero Padding) для визуализации.

    Физический смысл:
        Zero padding в k-пространстве эквивалентен sinc-интерполяции
        в пространственной области. Разрешение не улучшается, но
        визуально пиксели становятся мельче и форма РЛИ прорисовывается точнее.

    Args:
        E_final: финальная матрица в k-space после оконной функции, shape (M, N).
        target_shape: целевой размер (M_pad, N_pad). Если None — вычисляется через factor.
        factor: коэффициент увеличения (по умолчанию 2).

    Returns:
        E_padded: дополненная матрица, shape (M_pad, N_pad).
        pad_info: dict с информацией о padding (начальные индексы).
    """
    M, N = E_final.shape

    if target_shape is not None:
        M_pad, N_pad = target_shape
    else:
        M_pad = int(2 ** np.ceil(np.log2(M * factor)))
        N_pad = int(2 ** np.ceil(np.log2(N * factor)))
        M_pad = max(M_pad, M)
        N_pad = max(N_pad, N)

    E_padded = np.zeros((M_pad, N_pad), dtype=complex)

    # Размещаем данные в центре
    start_m = (M_pad - M) // 2
    start_n = (N_pad - N) // 2
    E_padded[start_m:start_m + M, start_n:start_n + N] = E_final

    return E_padded, {"start_m": start_m, "start_n": start_n, "M": M, "N": N}


def process_blind_zones(E_cart, kx_axis, ky_axis, window_type="taylor",
                        apply_zero_pad=True, pad_factor=2, **window_kwargs):
    """Полная обработка слепых зон (Этап 5, все шаги).

    Объединяет:
        1. Создание маски данных (Шаг 5.2)
        2. Применение оконной функции (Шаг 5.3)
        3. Жесткое обнуление слепых зон (Шаг 5.3, финал)
        4. Zero padding (Шаг 5.4)

    Args:
        E_cart: интерполированная матрица, shape (M_new, N_new).
        kx_axis: ось kx, shape (M_new,).
        ky_axis: ось ky, shape (N_new,).
        window_type: тип окна ("taylor", "hamming", "hann").
        apply_zero_pad: применять ли zero padding.
        pad_factor: коэффициент zero padding.
        **window_kwargs: параметры окна (nbar, sll для Тейлора).

    Returns:
        dict с ключами:
            E_final: финальная матрица после всех обработок.
            mask: бинарная маска.
            W_2D: 2D окно.
            E_windowed: матрица после окна (до обнуления).
            pad_info: информация о zero padding (если применялся).
    """
    # Шаг 5.2: Маска данных
    mask = create_data_mask(E_cart)

    # Шаг 5.3: Оконная функция
    E_windowed, W_2D = apply_window_2d(E_cart, window_type=window_type, **window_kwargs)

    # Шаг 5.3 (финал): Жесткое обнуление слепых зон
    E_final = E_windowed * mask

    # Шаг 5.4: Zero padding (опционально)
    pad_info = None
    if apply_zero_pad:
        E_final, pad_info = zero_pad_k_space(E_final, factor=pad_factor)

    return {
        "E_final": E_final,
        "mask": mask,
        "W_2D": W_2D,
        "E_windowed": E_windowed,
        "pad_info": pad_info,
    }


# ── Этап 6: Формирование изображения с полярным переформатированием ────────


def form_image_2d(E_k_space, dkx, dky):
    """Шаг 6.1-6.3: 2D IFFT + fftshift + физическое масштабирование осей.

    Физический смысл:
        2D IFFT переводит данные из k-пространства в пространственную область.
        fftshift центрирует нулевую дальность и нулевой допpler в центр матрицы.
        Физическое масштабирование переводит индексы пикселей в метры.

    Формулы (Оздемир):
        I = ifft2(E_k_space) * N
        X_span = 2π / Δkx
        Y_span = 2π / Δky

    Args:
        E_k_space: матрица в k-space, shape (M, N). Комплексная.
        dkx: шаг по оси kx (рад/м).
        dky: шаг по оси ky (рад/м).

    Returns:
        I_complex: комплексное изображение, shape (M, N).
        x_axis: ось дальности (м), shape (M,).
        y_axis: ось азимута (м), shape (N,).
        x_span: протяженность сцены по дальности (м).
        y_span: протяженность сцены по азимуту (м).
    """
    from scipy.fft import ifft2, fftshift

    M, N = E_k_space.shape

    # Шаг 6.1: 2D IFFT
    I_complex = ifft2(E_k_space)

    # Шаг 6.2: Центрирование (fftshift по обеим осям)
    I_complex = fftshift(I_complex)

    # Шаг 6.3: Физическое масштабирование осей
    x_span = 2.0 * np.pi / dkx
    y_span = 2.0 * np.pi / dky

    x_axis = (np.arange(M) - (M - 1) / 2.0) * (x_span / M)
    y_axis = (np.arange(N) - (N - 1) / 2.0) * (y_span / N)

    return I_complex, x_axis, y_axis, x_span, y_span


def compute_amplitude_db(I_complex, epsilon=1e-12):
    """Шаг 6.4: Вычисление амплитуды и перевод в дБ.

    Формулы:
        I_amp = |I_complex|
        I_dB = 20 * log10(I_amp + epsilon)
        I_normalized = I_dB - max(I_dB)

    Args:
        I_complex: комплексное изображение, shape (M, N).
        epsilon: малое число для защиты от log(0).

    Returns:
        amp_db_norm: нормализованная амплитуда в дБ, shape (M, N).
            Максимум = 0 дБ.
    """
    amp = np.abs(I_complex)
    amp_db = 20.0 * np.log10(amp + epsilon)
    amp_db_norm = amp_db - amp_db.max()
    return amp_db_norm


def compute_isar_reformatted(E, k, theta, grid_info, window_type="taylor",
                              pad_factor=2, **window_kwargs):
    """Полный конвейер Этапа 6: интерполяция + оконная функция + 2D IFFT.

    Объединяет Этапы 4-6 в один вызов.

    Args:
        E: исходная матрица эхо-сигнала, shape (M, N).
        k: вектор волновых чисел, shape (M,).
        theta: вектор углов, shape (N,).
        grid_info: dict из design_cartesian_grid().
        window_type: тип окна ("taylor", "hamming", "hann").
        pad_factor: коэффициент zero padding.
        **window_kwargs: параметры окна.

    Returns:
        dict с ключами:
            I_complex: комплексное изображение.
            amp_db: нормализованная амплитуда в дБ.
            x_axis, y_axis: физические оси (м).
            x_span, y_span: протяженность сцены (м).
            E_final: финальная матрица в k-space.
    """
    # Этап 4: Интерполяция
    E_cart = interpolate_polar_to_cartesian(E, k, theta, grid_info, method="linear")

    # Этап 5: Слепые зоны + оконная функция + zero padding
    kz = grid_info["kx_axis"]
    ky = grid_info["ky_axis"]
    blind = process_blind_zones(
        E_cart, kz, ky,
        window_type=window_type,
        apply_zero_pad=True,
        pad_factor=pad_factor,
        **window_kwargs,
    )
    E_final = blind["E_final"]

    # Этап 6: 2D IFFT + масштабирование
    dkx = grid_info["dkx"]
    dky = grid_info["dky"]
    I_complex, x_axis, y_axis, x_span, y_span = form_image_2d(E_final, dkx, dky)

    # Амплитуда в дБ
    amp_db = compute_amplitude_db(I_complex)

    return {
        "I_complex": I_complex,
        "amp_db": amp_db,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_span": x_span,
        "y_span": y_span,
        "E_final": E_final,
    }


def plot_three_way_comparison(amp_naive, extent_naive,
                               amp_nearest, extent_nearest,
                               amp_bilinear, extent_bilinear,
                               ax=None):
    """Шаг 6.6: Тройное сравнение РЛИ (наивное / nearest / bilinear+window).

    Args:
        amp_naive: амплитуда наивного метода (без переформатизации), dB.
        extent_naive:extent для imshow [xmin, xmax, ymax, ymin].
        amp_nearest: амплитуда nearest neighbor метода, dB.
        extent_nearest:extent для imshow.
        amp_bilinear: амплитуда bilinear+window метода, dB.
        extent_bilinear:extent для imshow.
        ax: массив из 3 matplotlib осей (если None — создаст свою фигуру).

    Returns:
        fig, axes: фигура и оси.
    """
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        axes = ax
        fig = axes[0].get_figure()

    titles = [
        "(A) No reformat (2D FFT) - arc smearing",
        "(B) Nearest Neighbor - aliasing artifacts",
        "(C) Bilinear + Taylor window - focused",
    ]
    amps = [amp_naive, amp_nearest, amp_bilinear]
    extents = [extent_naive, extent_nearest, extent_bilinear]

    for i, (title, amp, ext) in enumerate(zip(titles, amps, extents)):
        im = axes[i].imshow(amp, aspect="auto", cmap="jet",
                            extent=ext, vmin=-40, vmax=0)
        axes[i].set_xlabel("Cross-Range (m)")
        axes[i].set_ylabel("Range (m)")
        axes[i].set_title(title)
        plt.colorbar(im, ax=axes[i], label="dB")

    return fig, axes


# ── Этап 7: Интеграция, метрики и автоматическое переключение ──────────────


def should_use_polar_reformat(omega, pri, num_pulses, threshold_deg=3.0):
    """Шаг 7.2: Автоматическое определение необходимости полярной переформатизации.

    Физический смысл:
        При малых углах (Delta_theta < threshold) аппроксимация малых углов
        работает хорошо и полярная переформатизация не нужна.
        При больших углах — обязательна.

    Args:
        omega: угловая скорость (рад/с).
        pri: период повторения импульсов (с).
        num_pulses: число импульсов.
        threshold_deg: порог в градусах (по умолчанию 3°).

    Returns:
        use_polar: True если нужна полярная переформатизация.
        delta_theta_rad: суммарный угол в радианах.
        delta_theta_deg: суммарный угол в градусах.
    """
    T_obs = num_pulses * pri
    delta_theta_rad = omega * T_obs
    delta_theta_deg = math.degrees(delta_theta_rad)
    threshold_rad = math.radians(threshold_deg)
    use_polar = delta_theta_rad >= threshold_rad
    return use_polar, delta_theta_rad, delta_theta_deg


def compute_image_entropy(I_complex):
    """Шаг 7.4: Вычисление энтропии изображения.

    Физический смысл:
        Энтропия мера хаоса. Чем лучше сфокусировано изображение
        (энергия сконцентрирована в точках), тем ниже энтропия.
        Формула (Shannon entropy):
            H = -sum(p * log2(p))  где p = |I|^2 / sum(|I|^2)

    Args:
        I_complex: комплексное изображение, shape (M, N).

    Returns:
        entropy: значение энтропии (бит). Меньше = лучше фокусировка.
    """
    amp_sq = np.abs(I_complex) ** 2
    total = amp_sq.sum()
    if total < 1e-30:
        return 0.0
    p = amp_sq / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return entropy


def compute_isnr(I_focused, I_blurred):
    """Шаг 7.4: Вычисление отношения сигнал/шум (ISNR).

    Физический смысл:
        ISNR = 10 * log10( sum(|I_focused - I_blurred|^2) / sum(|I_blurred|^2) )
        Чем выше ISNR, тем лучше фокусировка.

    Args:
        I_focused: сфокусированное изображение (complex).
        I_blurred: размытое изображение (complex).

    Returns:
        isnr: ISNR в дБ.
    """
    noise = I_focused - I_blurred
    signal_power = np.sum(np.abs(noise) ** 2)
    noise_power = np.sum(np.abs(I_blurred) ** 2)
    if noise_power < 1e-30:
        return float("inf")
    isnr = 10.0 * np.log10(signal_power / noise_power)
    return isnr


def smart_pipeline(E, f_r, radar, target, omega, pri, num_pulses,
                   threshold_deg=3.0, Xmax_scene=50.0, Ymax_scene=50.0,
                   use_mocomp=True):
    """Шаг 7.2: Умный конвейер — автоматическое переключение режимов.

    Если Delta_theta >= threshold: полярная переформатизация + 2D IFFT.
    Если Delta_theta < threshold: стандартный 1D конвейер.

    Args:
        E: матрица сырых данных, shape (M, N).
        f_r: вектор частот, shape (M,).
        radar: объект Radar.
        target: объект Target.
        omega: угловая скорость (рад/с).
        pri: период повторения импульсов (с).
        num_pulses: число импульсов.
        threshold_deg: порог для переключения (градусы).
        Xmax_scene: размер сцены по дальности (м).
        Ymax_scene: размер сцены по азимуту (м).

    Returns:
        dict с результатами и флагом используемого метода.
    """
    use_polar, delta_theta_rad, delta_theta_deg = should_use_polar_reformat(
        omega, pri, num_pulses, threshold_deg
    )

    result = {
        "use_polar": use_polar,
        "delta_theta_rad": delta_theta_rad,
        "delta_theta_deg": delta_theta_deg,
        "method": "polar_reformat" if use_polar else "standard_1d",
    }

    if use_polar:
        # Новый конвейер: полярная переформатизация + 2D IFFT
        kspace = build_kspace_grid(f_r, omega, pri, num_pulses)
        k, theta = kspace["k"], kspace["theta"]
        grid = design_cartesian_grid(k, theta, Xmax_scene, Ymax_scene)
        isar = compute_isar_reformatted(
            E, k, theta, grid, window_type="taylor", pad_factor=2, nbar=4, sll=-35.0,
        )
        result.update(isar)
        result["x_axis"] = isar["x_axis"]
        result["y_axis"] = isar["y_axis"]
    else:
        # Стандартный конвейер: 1D + 1D
        from processing.range_compress import range_compress
        from processing.azimuth_compress import azimuth_compress, compute_amplitude
        from processing.mocomp import mocomp as mocomp_fn
        P, P_abs, range_axis, dr = range_compress(E, f_r, R0=target.R0)
        if use_mocomp:
            P, _, _, _ = mocomp_fn(P, range_axis)
        I, azimuth_axis, _ = azimuth_compress(
            P.astype(np.complex128), omega, radar.f_c, pri=pri
        )
        amp_db = compute_amplitude(I, mode="db")
        result["I_complex"] = I
        result["amp_db"] = amp_db
        result["x_axis"] = range_axis
        result["y_axis"] = azimuth_axis
        result["x_span"] = range_axis[-1] - range_axis[0]
        result["y_span"] = azimuth_axis[-1] - azimuth_axis[0]

    return result
