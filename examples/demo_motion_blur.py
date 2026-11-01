"""Демо 2: Размытие движением (Motion Blur) — физически валидный режим.

Сценарий: V!=0, omega!=0. MOCOMP отключен. Параметры из Плана фиксов (D1):
    N=256, PRI=5e-4 c, Δθ=3° за апертуру, V=10 м/с, α=0.3°, survey 40 м.

Показывает физику сырых данных:
  - косые линии на профиле дальности (Range Walk: 1.28 м = 4.3 бина за
    апертуру при PRI=5e-4 — без доплер-алиасинга: C1b 1.4×);
  - размытое РЛИ без MOCOMP (ход размывает как по дальности, так и по
    кросс-рейнджу).

Демо 3 (demo_full_pipeline_mocomp) использует ТОТ ЖЕ сценарий с включённым
MOCOMP — сравнение «до/после» честное (одна и та же цель, seed по умолчанию).

Границы: Δθ=3° формально превышает порог малого угла (C4 0.29×) — фактическое
размытие arc ~3 мм ≪ PSF 0.36 м (задокументировано); для метрологии больших
углов — полярный тракт (демо 4/5).
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

N_PULSES = 256
PRI = 5e-4
DELTA_THETA_DEG = 3.0
V_SC = 10.0          # м/с трансляции
ALPHA_DEG = 0.3      # угол направления движения, град
SURVEY = 40.0
R0 = 500000.0


def run(f_c=8e9, spectr_w=0.5e9, ang=DELTA_THETA_DEG, R0=R0,
        V=V_SC, alpha=ALPHA_DEG, omega=None,
        satellite="ICEsat2", verbose=True):
    omega = (omega if omega is not None
             else math.radians(DELTA_THETA_DEG) / (N_PULSES * PRI))
    pri = PRI

    radar = Radar(c=SPEED_OF_LIGHT, f_c=f_c, Xmax=SURVEY, Ymax=SURVEY,
                  spectr_w=spectr_w, ph_c=0.0, ang=9.0, nifft_size=256)

    filename = "matrices/matrices_%s_high.pkl" % satellite
    target = Target(filename, pri=pri, num_pulses=N_PULSES, R0=R0,
                    V=V, alpha=alpha, omega=omega)

    if verbose:
        print("=== Motion Blur (V=%.1f, alpha=%.2f°, Δθ=%.1f°, no MOCOMP) ==="
              % (V, alpha, math.degrees(omega * N_PULSES * pri)))
        print(format_report(check_applicability(radar, target)))

    E, f_r = generate_raw_matrix(radar, target)
    P, P_abs, range_axis, dr = range_compress(E, f_r, R0=R0)
    I, azimuth_axis, doppler_axis = azimuth_compress(
        P.astype(np.complex128), omega, f_c, pri=pri)
    amplitude = compute_amplitude(I, mode="linear")

    # Честные метрики (План фиксов §8.3): аналитический ход вместо
    # аргмакс-разности, которая измеряла заворот, а не дрейф
    walk_analytic = V * math.cos(math.radians(alpha)) * N_PULSES * pri  # м

    if verbose:
        print("  Range walk (аналитика): %.2f м = %.1f бина" % (walk_analytic, walk_analytic / dr))
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
    axes[1].set_title("ISAR Image (blurred without MOCOMP)")
    axes[1].set_xlabel("Cross-Range (m)")
    axes[1].set_ylabel("Range (m)")
    plt.colorbar(im1, ax=axes[1], label="Amplitude")

    def entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log(p + 1e-30))

    info_text = (
        "V = %.1f m/s\n"
        "alpha = %.2f deg\n"
        "range walk = %.2f m (%.1f bins)\n"
        "pri = %.1e s, Δθ = %.1f deg\n"
        "Entropy = %.2f"
    ) % (V, alpha, walk_analytic, walk_analytic / dr, pri,
         math.degrees(omega * N_PULSES * pri), entropy(amplitude))
    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes,
                 fontsize=12, verticalalignment="center", fontfamily="monospace")
    axes[2].set_title("Motion Parameters")
    axes[2].axis("off")

    fig.suptitle("Demo 2: Motion Blur (V=%.1f m/s, Δθ=%.1f°, no MOCOMP)"
                 % (V, math.degrees(omega * N_PULSES * pri)), fontsize=13)
    plt.tight_layout()
    plt.savefig("examples/demo_motion_blur.png", dpi=150)
    if sys.flags.interactive:
        plt.show()
    else:
        plt.close(fig)
        print("Saved: examples/demo_motion_blur.png")


if __name__ == "__main__":
    run(verbose=True)
