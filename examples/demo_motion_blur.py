"""Демо 2: Размытие движением (Motion Blur).

Сценарий: V!=0, omega!=0. MOCOMP отключен.
Демонстрирует физику сырых данных:
  - Косые линии на профиле дальности (Range Walk)
  - Полностью размытое РЛИ

Запуск:
    python -m examples.demo_motion_blur
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


def run(f_c=8e9, spectr_w=0.5e9, ang=9.0, R0=500000.0,
        V=50.0, alpha=0.3, omega=None,
        satellite="ICEsat2", verbose=True):
    if omega is None:
        omega = math.radians(ang) / 16.0

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=20.0, Ymax=20.0,
                  spectr_w=spectr_w, ph_c=0.0, ang=ang, nifft_size=1024)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=1.0, num_pulses=16, R0=R0, V=V, alpha=alpha, omega=omega)

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(P.astype(np.complex128), omega, f_c, pri=1.0)
    amplitude = compute_amplitude(I, mode="linear")

    peak_bins = np.argmax(P_abs, axis=0)
    drift = int(peak_bins[-1]) - int(peak_bins[0])

    if verbose:
        print("=== Motion Blur (V=%.1f, alpha=%.2f, w!=0, no MOCOMP) ===" % (V, alpha))
        print("  Range walk drift: %d bins" % drift)
        print("  Peak: %.1f" % amplitude.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    extent_range = [0, P_abs.shape[1] - 1, range_axis[-1], range_axis[0]]
    im0 = axes[0].imshow(P_abs, aspect="auto", cmap="plasma", extent=extent_range)
    axes[0].set_title("Range Profiles (diagonal lines = Range Walk)")
    axes[0].set_xlabel("Pulse #")
    axes[0].set_ylabel("Range (m)")
    plt.colorbar(im0, ax=axes[0], label="|P|")

    extent_isar = [azimuth_axis[0], azimuth_axis[-1], range_axis[-1], range_axis[0]]
    im1 = axes[1].imshow(amplitude, aspect="auto", cmap="jet", extent=extent_isar)
    axes[1].set_title("ISAR Image (blurred!)")
    axes[1].set_xlabel("Cross-Range (m)")
    axes[1].set_ylabel("Range (m)")
    plt.colorbar(im1, ax=axes[1], label="Amplitude")

    def entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log(p + 1e-30))

    info_text = (
        "V = %.1f m/s\n"
        "alpha = %.2f rad\n"
        "omega = %.4f rad/s\n"
        "Drift = %d bins\n"
        "Entropy = %.2f"
    ) % (V, alpha, omega, drift, entropy(amplitude))
    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes,
                 fontsize=12, verticalalignment="center", fontfamily="monospace")
    axes[2].set_title("Motion Parameters")
    axes[2].axis("off")

    fig.suptitle("Demo 2: Motion Blur (V!=0, w!=0, no MOCOMP)", fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_motion_blur.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_motion_blur.png")


if __name__ == "__main__":
    run(verbose=True)
