"""Демо 1: Идеальный разворот (Turntable Model).

Сценарий: V=0, omega!=0. MOCOMP отключен.
Демонстрирует, что базовая обработка работает идеально
при чистом вращении без поступательного движения.

Запуск:
    python -m examples.demo_ideal_turntable
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import matplotlib.pyplot as plt

from models.radar import Radar
from models.target import Target
from simulation.raw_generator import generate_raw_matrix
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude

SPEED_OF_LIGHT = 3e8


def run(f_c=8e9, spectr_w=0.5e9, ang=9.0, R0=500000.0, omega=None,
        satellite="ICEsat2", verbose=True):
    if omega is None:
        omega = math.radians(ang) / 16.0

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=1.0, num_pulses=16, R0=R0, V=0.0, alpha=0.0, omega=omega)

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(P.astype(np.complex128), omega, f_c, pri=1.0)
    amplitude = compute_amplitude(I, mode="linear")

    if verbose:
        print("=== Ideal Turntable (V=0, w!=0) ===")
        print("  Raw data E: %s" % str(E.shape))
        print("  Range profiles P: %s" % str(P.shape))
        print("  ISAR image I: %s" % str(I.shape))
        print("  Azimuth axis: [%.3f, %.3f] m" % (azimuth_axis[0], azimuth_axis[-1]))
        print("  Range axis: [%.1f, %.1f] m" % (range_axis[0], range_axis[-1]))
        print("  Peak: %.1f" % amplitude.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    extent_range = [0, P_abs.shape[1] - 1, range_axis[-1], range_axis[0]]
    axes[0].imshow(P_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0].set_title("Range Profiles")
    axes[0].set_xlabel("Pulse #")
    axes[0].set_ylabel("Range (m)")

    extent_isar = [azimuth_axis[0], azimuth_axis[-1], range_axis[-1], range_axis[0]]
    im = axes[1].imshow(amplitude, aspect="auto", cmap="jet", extent=extent_isar)
    axes[1].set_title("ISAR Image (linear)")
    axes[1].set_xlabel("Cross-Range (m)")
    axes[1].set_ylabel("Range (m)")
    plt.colorbar(im, ax=axes[1], label="Amplitude")

    amp_db = compute_amplitude(I, mode="db")
    im2 = axes[2].imshow(amp_db, aspect="auto", cmap="jet", extent=extent_isar, vmin=-30, vmax=0)
    axes[2].set_title("ISAR Image (dB)")
    axes[2].set_xlabel("Cross-Range (m)")
    axes[2].set_ylabel("Range (m)")
    plt.colorbar(im2, ax=axes[2], label="Amplitude (dB)")

    fig.suptitle("Demo 1: Ideal Turntable (V=0, w!=0, no MOCOMP)", fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_ideal_turntable.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_ideal_turntable.png")


if __name__ == "__main__":
    run(verbose=True)
