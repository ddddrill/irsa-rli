"""Оконные функции взвешивания (дальность/азимут).

Единая точка определений имён и функций окон для GUI и обоих конвейеров
(стандартного и полярного). Взвешивание перед IFFT/FFT — C&M, §3.3;
компромисс «динамический диапазон / разрешение / SNR» — C&M, §6.3.1.

Нормировка: все окна возвращаются как w / mean(w) — восстанавливает
когерентное усиление (амплитуда пиков сопоставима между окнами).
Для "none" (boxcar) нормировка не применяется: mean(w) == 1, возвращаются
единицы без вычислений — гарантия битовой идентичности старому коду.
"""
import numpy as np
from scipy.signal.windows import chebwin

# Имена окон в порядке GUI. "none" = boxcar (честная PSF ядра Дирихле).
WINDOW_NAMES = ["none", "hamming", "hann", "kaiser", "chebwin", "taylor"]


def _boxcar(N, param=None):
    return np.ones(N)


def _hann(N, param=None):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1)))


def _hamming(N, param=None):
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N) / (N - 1))


def _kaiser(N, param=None):
    # Исчерпывающее семейство: beta = 0 -> boxcar, рост beta -> ниже SLL,
    # шире главный лепесток. Официальное соответствие scipy: kaiser(N, 8.6)
    # ~= blackman(N). Формула окна — C&M, §6.3.1.
    beta = 8.0 if param is None else float(param)
    return np.kaiser(N, beta)


def _chebwin(N, param=None):
    # Dolph-Chebyshev: минимаксно-оптимальное окно при заданной ширине ГЛ,
    # боковые лепестки равноуровневые (equiripple). at — уровень в дБ.
    # Радарный стандарт 60 дБ. scipy вырождается при N < 4 -> boxcar.
    if N < 4:
        return np.ones(N)
    at = 60.0 if param is None else float(param)
    w = chebwin(N, at=at)
    if not np.all(np.isfinite(w)):
        return np.ones(N)
    return w


def _taylor(N, param=None):
    # Taylor: первые nbar-1 боковых на уровне sll, далее спад (компромисс
    # Чебышёв/Хэнн). Переиспользуем taylor_window_1d полярного конвейера —
    # импорт внутри функции, чтобы не создавать цикл (polar_reformat
    # загружается до windows при импорте data_processor).
    from processing.polar_reformat import taylor_window_1d
    if isinstance(param, dict):
        nbar = param.get("nbar", 4)
        sll = param.get("sll", -35.0)
    else:
        nbar, sll = 4, -35.0
    return taylor_window_1d(N, nbar=nbar, sll=sll)


# name или None -> функция окна (N, param=None). param:
#   kaiser -> beta (дефолт 8.0); chebwin -> at, дБ (дефолт 60.0);
#   taylor -> dict(nbar=4, sll=-35.0); остальные окна param игнорируют.
WINDOW_FUNCTIONS = {
    "none": _boxcar,
    None: _boxcar,
    "hamming": _hamming,
    "hann": _hann,
    "kaiser": _kaiser,
    "chebwin": _chebwin,
    "taylor": _taylor,
}


def get_weighting(name, N, param=None):
    """Нормированное 1D окно взвешивания длины N.

    Args:
        name: имя окна ("none", "hamming", "hann", "kaiser", "chebwin",
            "taylor") или None. Неизвестное имя -> boxcar.
        N: длина окна (число отсчётов).
        param: параметр окна (см. WINDOW_FUNCTIONS); None = дефолт.

    Returns:
        w: вещественный массив shape (N,), нормированный на w / mean(w).
            Для "none" возвращаются единицы без нормировки.
    """
    N = int(N)
    if not name or name == "none":
        return np.ones(N)
    win = WINDOW_FUNCTIONS.get(name, _boxcar)(N, param)
    win = np.asarray(win, dtype=float)
    return win / win.mean()
