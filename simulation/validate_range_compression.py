"""Визуальная валидация Этапа 4: сжатие по дальности (Range Compression).

Строит 2D-карту профилей дальности |P(range, slow_time)|.

Ось X — дальность (метры, относительно R0).
Ось Y — номер импульса (медленное время).
Цвет  — амплитуда сигнала.

Ожидаемый результат:
  - Если есть поступательное движение (V != 0): косые линии (Range Walk).
  - Если чистое вращение (V = 0): горизонтальные линии.
  - Если шум без структуры: ошибка в фазовой формуле.

Запуск:
    python -m simulation.validate_range_compression
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress

SPEED_OF_LIGHT = 3e8


def plot_range_profiles(P, range_axis, title="Range Compression"):
    """Построить 2D-график профилей дальности."""
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [range_axis[0], range_axis[-1], P.shape[1] - 1, 0]
    im = ax.imshow(
        P,
        aspect="auto",
        extent=extent,
        cmap="plasma",
        interpolation="nearest",
    )
    ax.set_xlabel("Дальность, м (относительно R0)")
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
         "label": "Test 1: Pure rotation (V=0, w!=0)"},
        {"V": 100.0, "alpha": 0.0, "omega": 0.0,
         "label": "Test 2: Pure translation (V=100, w=0)"},
        {"V": 50.0, "alpha": 0.3, "omega": omega,
         "label": "Test 3: Combined (V=50, w!=0)"},
    ]

    figs = []
    for sc in scenarios:
        target = Target(
            "matrices/matrices_ICEsat2_high.pkl",
            pri=1.0, num_pulses=16,
            R0=R0, V=sc["V"], alpha=sc["alpha"], omega=sc["omega"],
        )
        E, f_r = generate_raw_matrix(radar, target)
        P, range_axis, dr = range_compress(E, f_r, R0=R0)
        fig, ax = plot_range_profiles(P, range_axis, title=sc["label"])
        figs.append(fig)
        plt.close(fig)

        peak_idx = np.argmax(P, axis=0)
        drift = int(peak_idx[-1]) - int(peak_idx[0])
        print(sc["label"])
        print("  dr = %.3f m, range axis: [%.1f, %.1f] m" % (dr, range_axis[0], range_axis[-1]))
        print("  Peak indices: %s" % peak_idx.tolist())
        print("  Drift: %d bins" % drift)
        print()

    for i, fig in enumerate(figs):
        fig.savefig("simulation/test_%d_range_compression.png" % (i + 1), dpi=150)
        print("Saved: simulation/test_%d_range_compression.png" % (i + 1))

    plt.show()


if __name__ == "__main__":
    run_validation()
