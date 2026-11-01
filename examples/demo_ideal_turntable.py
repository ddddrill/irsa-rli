"""Демо 1: Идеальный разворот (Turntable Model) — физически валидный режим.

Сценарий: V=0, omega!=0. MOCOMP отключен. Параметры из Плана фиксов (D1):
    N=256, PRI=0.05 c, Δθ=0.8° за апертуру, survey 40 м, R0=500 км.
Конфигурация валидна по guard'ам: C1a 14.5×, C1b 14.3×, C2 1.67×, C4 1.10×
(прежний режим N=16/PRI=1/Δθ=9° был в алиасинге ×12.4 — позиции были ложными).
Цена честного малого угла стандартного тракта: PSF по кросс-рейнджу
λ/(2Δθ)·0.886 = 1.19 м; большие углы — демо 4/5 (полярный тракт).

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
from simulation.applicability import check_applicability, format_report
from processing.range_compress import range_compress
from processing.azimuth_compress import azimuth_compress, compute_amplitude

SPEED_OF_LIGHT = 3e8

# Параметры валидного сценария (План фиксов §8.2, таблица «демо 1»)
N_PULSES = 256
PRI = 0.05
DELTA_THETA_DEG = 0.8
SURVEY = 40.0
R0 = 500000.0


def run(f_c=8e9, spectr_w=0.5e9, ang=DELTA_THETA_DEG, R0=R0, omega=None,
        satellite="solarB", verbose=True):
    omega = (omega if omega is not None
             else math.radians(DELTA_THETA_DEG) / (N_PULSES * PRI))

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=SURVEY, Ymax=SURVEY,
                  spectr_w=spectr_w, ph_c=0.0, ang=9.0, nifft_size=256)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=PRI, num_pulses=N_PULSES, R0=R0,
                    V=0.0, alpha=0.0, omega=omega)

    if verbose:
        print("=== Ideal Turntable (V=0, Δθ=%.1f°) ===" % (
            math.degrees(omega * N_PULSES * PRI),))
        print(format_report(check_applicability(radar, target)))

        # Топ-5 истинных позиций рассеивателей (для сравнения с пиками РЛИ:
        # PSF_x = 0.886·c/2B, PSF_y = 0.886·λ/(2Δθ); у макс y0 теперь
        # largest интенсивности смежные позиции могут сливаться)
        sc = target._load_scatterers()
        top = np.argsort(sc["intens"])[::-1][:5]
        psf_x = 0.886 * SPEED_OF_LIGHT / (2 * spectr_w)
        psf_y = 0.886 * (SPEED_OF_LIGHT / f_c) / (2 * omega * N_PULSES * PRI)
        print(f"  PSF: dx={psf_x:.3f} м, dy={psf_y:.2f} м (допуск V1 = 2·PSF)")
        print("  Топ-5 истинных позиций (x, y), интенсивность:")
        for i in top:
            print(f"    ({sc['x0'][i]:+.2f}, {sc['y0'][i]:+.2f}) м,"
                  f" σ={sc['intens'][i]:.0f}")

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(
        P.astype(np.complex128), omega, f_c, pri=PRI)
    amplitude = compute_amplitude(I, mode="linear")

    if verbose:
        print("  Raw data E: %s" % str(E.shape))
        print("  Range profiles P: %s" % str(P.shape))
        print("  ISAR image I: %s" % str(I.shape))
        print("  Azimuth axis: [%.3f, %.3f] м" % (azimuth_axis[0], azimuth_axis[-1]))
        print("  Range axis: [%.1f, %.1f] м" % (range_axis[0], range_axis[-1]))
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

    fig.suptitle("Demo 1: Ideal Turntable (V=0, Δθ=%.1f°, валидный режим)"
                 % math.degrees(omega * N_PULSES * PRI), fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_ideal_turntable.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_ideal_turntable.png")


if __name__ == "__main__":
    run(verbose=True)
