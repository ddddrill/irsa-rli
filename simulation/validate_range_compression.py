"""Визуальная валидация Этапов 2 и 3: сжатие по дальности (Range Compression).

Применяет одномерное ОБПФ (IFFT) по оси частот к матрице сырых данных E
и строит 2D-график профилей дальности.

Ось X — бины дальности (индексы после IFFT).
Ось Y — номер импульса (медленное время).

Ожидаемый результат:
  - Если есть поступательное движение (V != 0): косые линии (Range Walk).
  - Если чистое вращение (V = 0): горизонтальные линии.
  - Если шум без структуры: ошибка в фазовой формуле.

Запуск:
    python -m simulation.validate_range_compression
"""
import math
import numpy as np
from scipy.fft import ifft, fftshift
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix

SPEED_OF_LIGHT = 3e8


def range_compression(E, f_r):
    """Сжатие по дальности: IFFT по оси частот (строкам).

    Args:
        E: матрица сырых данных M×N (частоты × импульсы).
        f_r: вектор частот (M,).

    Returns:
        rc: матрица профилей дальности M×N (модуль после IFFT).
        range_axis: ось дальностей (метры).
    """
    M, N = E.shape
    rc_raw = ifft(E, axis=0)
    rc = np.abs(fftshift(rc_raw, axes=0))

    df = f_r[1] - f_r[0]
    fz = M * df
    dt = 1.0 / fz
    t = np.arange(M) * dt
    range_axis = (t * SPEED_OF_LIGHT) / 2.0
    range_axis = range_axis - range_axis[M - 1] / 2.0

    return rc, range_axis


def plot_range_profiles(rc, range_axis, title="Range Compression"):
    """Построить 2D-график профилей дальности."""
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [range_axis[0], range_axis[-1], rc.shape[1] - 1, 0]
    im = ax.imshow(
        rc,
        aspect="auto",
        extent=extent,
        cmap="plasma",
        interpolation="nearest",
    )
    ax.set_xlabel("Дальность, м")
    ax.set_ylabel("Номер импульса (медленное время)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Амплитуда")
    plt.tight_layout()
    return fig, ax


def run_validation():
    f_c = 8e9
    spectr_w = 0.5e9
    ang = 9.0
    R0 = 500000.0
    omega = math.radians(ang) / 16.0

    radar = Radar(
        c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
        spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024,
    )

    scenarios = [
        {"V": 0.0, "alpha": 0.0, "omega": omega,
         "label": "Тест 1: Чистое вращение (V=0, w!=0) → горизонтальные линии"},
        {"V": 100.0, "alpha": 0.0, "omega": 0.0,
         "label": "Тест 2: Чистое поступательное (V=100, w=0) → косые линии"},
        {"V": 50.0, "alpha": 0.3, "omega": omega,
         "label": "Тест 3: Комбинированное (V=50, w!=0) → косые + вращение"},
    ]

    figs = []
    for sc in scenarios:
        target = Target(
            "matrices/matrices_ICEsat2_high.pkl",
            pri=1.0, num_pulses=16,
            R0=R0, V=sc["V"], alpha=sc["alpha"], omega=sc["omega"],
        )
        E, f_r = generate_raw_matrix(radar, target)
        rc, range_axis = range_compression(E, f_r)
        fig, ax = plot_range_profiles(rc, range_axis, title=sc["label"])
        figs.append(fig)
        plt.close(fig)

        peak_idx = np.argmax(rc, axis=0)
        print(f'{sc["label"]}')
        print(f'  Peak indices: {peak_idx.tolist()}')
        print(f'  Drift: {peak_idx[-1] - peak_idx[0]} bins')
        print()

    for i, fig in enumerate(figs):
        fig.savefig(f"simulation/test_{i+1}_range_compression.png", dpi=150)
        print(f"Saved: simulation/test_{i+1}_range_compression.png")

    plt.show()


if __name__ == "__main__":
    run_validation()
